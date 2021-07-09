__author__ = "Yuyu Luo"

import re
import json

import torch

def get_candidate_columns(src):
    col_list = re.findall('<col>.*</col>', src)[0].lower().split(' ')
    return col_list[1:-1] # remove <col> </col>


def get_template(src):
    col_list = re.findall('<template>.*</template>', src)[0].lower().split(' ')
    return col_list[1:-1] # remove <template> </template>

def get_all_table_columns(data_file):
    with open(data_file, 'r') as fp:
        data = json.load(fp)
    '''
    return:
    {'chinook_1': {'Album': ['AlbumId', 'Title', 'ArtistId'],
      'Artist': ['ArtistId', 'Name'],
      'Customer': ['CustomerId',
       'FirstName',
    '''
    return data

def get_y_from_aggy(aggy):
    if '(' in aggy and ')' in aggy:
        return aggy[:aggy.find("(")], aggy[aggy.find("(")+1:aggy.find(")")]
    else:
        return None, aggy

def guide_decoder_by_candidates(trg_field, input_source, table_columns, topk_ids, topk_tokens, current_token_type, pred_tokens_list):
    '''
    get the current token types (X, Y,...),
    we use the topk tokens from the decoder and the candidate columns to inference the "best" pred_token.
    table_columns: all columns in this table.
    topk_tokens: the top-k candidate predicted tokens
    current_token_type = x|y|groupby-axis|bin x|  if_template:[orderby-axis, order-type, chart_type]
    pred_tokens_list: the predicted tokens
    '''
    # candidate columns mentioned by the NL query
    candidate_columns = get_candidate_columns(input_source)

    best_token = topk_tokens[0]
    best_id = topk_ids[0]

#     print('candidate_columns:', candidate_columns)
#     print('table_columns:', table_columns)
#     print('topk_tokens:', topk_tokens)
#     print('current_token_type:', current_token_type)

    if current_token_type in ['x_axis', 'groupby_axis']:
        #         print('Case-1')
        if best_token not in table_columns:
            # correct:
            #             print('Case-1->correct1')
            is_in_topk = False
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    best_token = tok
                    best_id = trg_field.vocab.stoi[best_token]
                    is_in_topk = True
                    break
            if is_in_topk == False and len(candidate_columns) > 0:
                best_token = candidate_columns[0]
                best_id = trg_field.vocab.stoi[best_token]

        if best_token in table_columns and best_token not in candidate_columns:
            # correct:
            #             print('Case-1->correct2')
            for tok in topk_tokens:
                if tok in candidate_columns and tok in table_columns:
                    best_token = tok
                    best_id = trg_field.vocab.stoi[best_token]
                    is_in_topk = True
                    break

    if current_token_type == 'y_axis':
        #         print('Case-2')
        # get the y from the possible agg(y)
        agg, y = get_y_from_aggy(best_token)
        if y != '*':
            if y not in table_columns:
                # correct:
                #                 print('Case-2->correct1')
                is_in_topk = False
                for tok in topk_tokens:
                    agg, tok = get_y_from_aggy(tok)
                    if tok in candidate_columns and tok in table_columns:
                        if agg != None:
                            best_token = agg+'('+tok+')'
                            best_id = trg_field.vocab.stoi[best_token]
                        else:
                            best_token = tok
                            best_id = trg_field.vocab.stoi[best_token]
#                         print('y-best_token:', best_token)
                        is_in_topk = True
                        break
                if is_in_topk == False and len(candidate_columns) > 0:
                    agg, tok = get_y_from_aggy(topk_tokens[0])
                    if pred_tokens_list[pred_tokens_list.index('visualize') + 1] == 'scatter':
                        if pred_tokens_list[pred_tokens_list.index('select') + 1] == candidate_columns[0]:
                            if agg != None:
                                if len(candidate_columns) > 1:
                                    best_token = agg + '(' + candidate_columns[1] + ')'
                                else:
                                    best_token = agg + '(' + candidate_columns[0] + ')'
                                best_id = trg_field.vocab.stoi[best_token]
                            else:
                                if len(candidate_columns) > 1:
                                    best_token = candidate_columns[1]
                                else:
                                    best_token = candidate_columns[0]
                                best_id = trg_field.vocab.stoi[best_token]
                    else:
                        if agg != None:
                            best_token = agg+'('+candidate_columns[0]+')'
                            best_id = trg_field.vocab.stoi[best_token]
                        else:
                            best_token = candidate_columns[0]
                            best_id = trg_field.vocab.stoi[best_token]

            # if y in table_columns and y not in candidate_columns:
            #     for tok in topk_tokens:
            #         agg, tok = get_y_from_aggy(tok)
            #         if tok in candidate_columns and tok in table_columns:
            #             if agg != None:
            #                 best_token = agg+'('+tok+')'
            #                 best_id = trg_field.vocab.stoi[best_token]
            #             else:
            #                 best_token = tok
            #                 best_id = trg_field.vocab.stoi[best_token]
            #             is_in_topk = True
            #             break

    if current_token_type == 'bin_axis':  # bin [x] by ..
        best_token = pred_tokens_list[pred_tokens_list.index('select')+1]
        best_id = trg_field.vocab.stoi[best_token]

    template_list = get_template(input_source)
    if '[c]' not in template_list:  # have the chart template
        if current_token_type == 'chart_type':
            #             print('Case-4')
            best_token = template_list[template_list.index('visualize') + 1]
            best_id = trg_field.vocab.stoi[best_token]

        if current_token_type == 'orderby_axis':
            #             print('Case-3')
            if template_list[template_list.index('order')+2] == '[x]':
                best_token = pred_tokens_list[pred_tokens_list.index(
                    'select')+1]
                best_id = trg_field.vocab.stoi[best_token]

            if template_list[template_list.index('order')+2] == '[y]':
                best_token = pred_tokens_list[pred_tokens_list.index('from')-1]
                best_id = trg_field.vocab.stoi[best_token]

        if current_token_type == 'orderby_type':
            #             print('Case-5')
            best_token = template_list[template_list.index('order') + 3]
            best_id = trg_field.vocab.stoi[best_token]

#     print('the output token:', best_token,'\n')
    return best_id, best_token


def translate_sentence(sentence, src_field, trg_field, TOK_TYPES, tok_types, model, device, max_len=128):
    model.eval()

    # process the tok_type
    if isinstance(tok_types, str):
        tok_types_ids = tok_types.lower().split(' ')
    else:
        tok_types_ids = [tok_type.lower() for tok_type in tok_types]
    tok_types_ids = [TOK_TYPES.init_token] + tok_types_ids + [TOK_TYPES.eos_token]
    tok_types_ids_indexes = [TOK_TYPES.vocab.stoi[tok_types_id] for tok_types_id in tok_types_ids]
    tok_types_tensor = torch.LongTensor(tok_types_ids_indexes).unsqueeze(0).to(device)

    if isinstance(sentence, str):
        tokens = sentence.lower().split(' ')
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    # visibility matrix
    batch_visibility_matrix = model.make_visibility_matrix(src_tensor, src_field)

    with torch.no_grad():
        enc_src, enc_attention = model.encoder(src_tensor, src_mask, tok_types_tensor, batch_visibility_matrix)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        pred_token = output.argmax(2)[:, -1].item()

        trg_indexes.append(pred_token)

        if pred_token == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    trg_tokens = [trg_field.vocab.itos[i] for i in trg_indexes]

    return trg_tokens[1:], attention, enc_attention


def translate_sentence_with_guidance(db_id, table_id, sentence, src_field, trg_field,
                                     TOK_TYPES, tok_types, SRC, model, db_tables_columns,
                                     device, max_len=128, show_progress = False):
    model.eval()

    # process the tok_type
    if isinstance(tok_types, str):
        tok_types_ids = tok_types.lower().split(' ')
    else:
        tok_types_ids = [tok_type.lower() for tok_type in tok_types]
    tok_types_ids = [TOK_TYPES.init_token] + \
                    tok_types_ids + [TOK_TYPES.eos_token]
    tok_types_ids_indexes = [TOK_TYPES.vocab.stoi[tok_types_id]
                             for tok_types_id in tok_types_ids]
    tok_types_tensor = torch.LongTensor(
        tok_types_ids_indexes).unsqueeze(0).to(device)

    if isinstance(sentence, str):
        tokens = sentence.lower().split(' ')
    else:
        tokens = [token.lower() for token in sentence]

    tokens = [src_field.init_token] + tokens + [src_field.eos_token]

    src_indexes = [src_field.vocab.stoi[token] for token in tokens]

    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)

    src_mask = model.make_src_mask(src_tensor)

    # visibility matrix
    batch_visibility_matrix = model.make_visibility_matrix(src_tensor, SRC)

    with torch.no_grad():
        enc_src, enc_attention = model.encoder(src_tensor, src_mask,
                                               tok_types_tensor, batch_visibility_matrix)

    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]
    trg_tokens = []

    current_token_type = None
    if show_progress == True:
        print('Show the details in each tokens:')
    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        table_columns = []
        try:  # get all columns in a table
            table_columns = db_tables_columns[db_id][table_id]
        except:
            print('[Fail] get all columns in a table')
            table_columns = []

        if current_token_type != 'table_name':
            # get top-3 candidates tokens
            topk_ids = torch.topk(output, k=3, dim=2, sorted=True).indices[:, -1, :].tolist()[0]
            topk_tokens = [trg_field.vocab.itos[tok_id] for tok_id in topk_ids]

            '''
            apply guide_decoder_by_candidates
            '''
            pred_id, pred_token = guide_decoder_by_candidates(
                trg_field, sentence, table_columns, topk_ids,
                topk_tokens, current_token_type, trg_tokens
            )

            if show_progress == True:
                if current_token_type == None:
                    print('-------------------\nCurrent Token Type: Query Sketch Part , top-3 tokens: [{}]'.format(', '.join(topk_tokens)))
                else:
                    print('-------------------\nCurrent Token Type: {} , original top-3 tokens: [{}] , the final tokens by VisAwareTranslation: {}'.format(current_token_type, ', '.join(topk_tokens), pred_token))

        else:
            # prediction the table name, note that the following codes are designed for single table !!!
            pred_token = table_id
            pred_id = trg_field.vocab.stoi[pred_token]
            if show_progress == True:
                print('-------------------\nCurrent Token Type: Table Name , top-3 tokens: [{}]'.format(current_token_type, pred_token))


        current_token_type = None

        trg_indexes.append(pred_id)
        trg_tokens.append(pred_token)

        # update the current_token_type and pred_axis here
        if i == 0:
            current_token_type = 'chart_type'

        if i > 1:
            if trg_tokens[-1] == 'select' and trg_tokens[-2] in ['bar', 'pie', 'line', 'scatter']:
                current_token_type = 'x_axis'

            if trg_tokens[-1] == 'bin':
                current_token_type = 'bin_axis'

        if i > 2:
            if trg_tokens[-1] == ',' and trg_tokens[-3] == 'select':
                current_token_type = 'y_axis'

            if trg_tokens[-1] == 'from' and trg_tokens[-3] == ',':
                current_token_type = 'table_name'

            if trg_tokens[-2] == 'group' and trg_tokens[-1] == 'by':
                current_token_type = 'groupby_axis'

            if trg_tokens[-2] == 'order' and trg_tokens[-1] == 'by':
                current_token_type = 'orderby_axis'

            if trg_tokens[-3] == 'order' and trg_tokens[-2] == 'by':
                current_token_type = 'orderby_type'

        if pred_id == trg_field.vocab.stoi[trg_field.eos_token]:
            break

    return trg_tokens, attention, enc_attention


def postprocessing_group(gold_q_tok, pred_q_tok):
    # 2. checking (and correct) group-by
    # rule 1: COUNT(*) and COUNT(Y-axis) should have the same results
    if 'group' in gold_q_tok and 'group' in pred_q_tok:
        # rule 1
        gold_agg_y = gold_q_tok[gold_q_tok.index('from') - 1]
        pred_agg_y = pred_q_tok[pred_q_tok.index('from') - 1]
        if 'count' in gold_agg_y and 'count' in pred_agg_y:
            gold_y = re.findall('count(.*)', gold_agg_y)[0].replace('(', '').replace(')', '')
            pred_y = re.findall('count(.*)', pred_agg_y)[0].replace('(', '').replace(')', '')
            gold_x = gold_q_tok[gold_q_tok.index('select') + 1]
            pred_x = pred_q_tok[pred_q_tok.index('select') + 1]
            if gold_y == pred_y:
                pass
            else:
                if gold_x == pred_x and (gold_y == '*' or pred_y == '*'):
                    pred_q_tok = [gold_agg_y if x == pred_agg_y else x for x in pred_q_tok]

    # rule 2: if other part is the same, and only add group-by part, the result should be the same
    if 'group' not in gold_q_tok and 'group' in pred_q_tok:
        groupby_x = pred_q_tok[pred_q_tok.index('group') + 2]
        if ''.join(pred_q_tok).replace('groupby' + groupby_x, '') == ''.join(gold_q_tok):
            pred_q_tok = gold_q_tok

    return pred_q_tok


def postprocessing(gold_query, pred_query, if_template, src_input):
    try:
        # get the template:
        chart_template = re.findall('<template>.*</template>', src_input)[0]
        chart_template_tok = chart_template.lower().split(' ')

        gold_q_tok = gold_query.lower().split(' ')
        pred_q_tok = pred_query.lower().split(' ')

        # 0. visualize type. if we have the template, the visualization type must be matched.
        if if_template:
            pred_q_tok[pred_q_tok.index('visualize') + 1] = gold_q_tok[gold_q_tok.index('visualize') + 1]

        # 1. Table Checking. If we focus on single table, must match!!!
        if 'from' in pred_q_tok and 'from' in gold_q_tok:
            pred_q_tok[pred_q_tok.index('from') + 1] = gold_q_tok[gold_q_tok.index('from') + 1]

        pred_q_tok = postprocessing_group(gold_q_tok, pred_q_tok)

        # 3. Order-by. if we have the template, we can checking (and correct) the predicting order-by
        # rule 1: if have the template, order by [x]/[y], trust to the select [x]/[y]
        if 'order' in gold_q_tok and 'order' in pred_q_tok and if_template:
            order_by_which_axis = chart_template_tok[chart_template_tok.index('order') + 2]  # [x], [y], or [o]
            x = pred_q_tok[pred_q_tok.index('select') + 1]
            y = pred_q_tok[pred_q_tok.index('from') - 1]
            if order_by_which_axis == '[x]':
                pred_q_tok[pred_q_tok.index('order') + 2] = x
            elif order_by_which_axis == '[y]':
                pred_q_tok[pred_q_tok.index('order') + 2] = y
            else:
                pass

        elif 'order' in gold_q_tok and 'order' not in pred_q_tok and if_template:
            order_by_which_axis = chart_template_tok[chart_template_tok.index('order') + 2]  # [x], [y], or [o]
            order_type = chart_template_tok[chart_template_tok.index('order') + 3]
            x = pred_q_tok[pred_q_tok.index('select') + 1]
            y = pred_q_tok[pred_q_tok.index('from') - 1]

            if x == gold_q_tok[gold_q_tok.index('order') + 2] or y == gold_q_tok[gold_q_tok.index('order') + 2]:
                pred_q_tok += ['order', 'by', gold_q_tok[gold_q_tok.index('order') + 2]]
                if gold_q_tok.index('order') + 3 < len(gold_q_tok):
                    pred_q_tok += [gold_q_tok[gold_q_tok.index('order') + 3]]
            else:
                pass

        else:
            pass

        pred_q_tok = postprocessing_group(gold_q_tok, pred_q_tok)

        # 4. checking (and correct) biniing
        # rule 1: [interval] bin 假设BIN这个问题，我们能解决，因为现在BIN有点乱
        # rule 2: bin by [x]
        if 'bin' in gold_q_tok and 'bin' in pred_q_tok:
            # rule 1
            if_bin_gold, if_bin_pred = False, False

            for binn in ['by time', 'by year', 'by weekday']:
                if binn in gold_query:
                    if_bin_gold = binn
                if binn in pred_query:
                    if_bin_pred = binn

            if (if_bin_gold != False and if_bin_pred != False) and (if_bin_gold != if_bin_pred):
                pred_q_tok[pred_q_tok.index('bin') + 3] = if_bin_gold.replace('by ', '')

            # rule 2
            x = pred_q_tok[pred_q_tok.index('select') + 1]
            bin_by_which_axis = chart_template_tok[chart_template_tok.index('bin') + 1]  # [x]
            if bin_by_which_axis == '[x]':
                pred_q_tok[pred_q_tok.index('bin') + 1] = x

        if 'bin' in gold_q_tok and 'bin' not in pred_q_tok and 'group' in pred_q_tok:
            # rule 3: group-by x and bin x by time in the bar chart should be the same.
            bin_x = gold_q_tok[gold_q_tok.index('bin') + 1]
            group_x = pred_q_tok[pred_q_tok.index('group') + 2]
            if bin_x == group_x:
                if ''.join(pred_q_tok).replace('groupby' + group_x, '') == ''.join(gold_q_tok).replace(
                        'bin' + bin_x + 'bytime', ''):
                    pred_q_tok = gold_q_tok
    except:
        print('error at post processing')
    return ' '.join(pred_q_tok)