__author__ = "Yuyu Luo"

'''
This script handles the testing process.
'''


import torch
import torch.nn as nn

from utilities.inference import translate_sentence_with_guidance, postprocessing, get_all_table_columns
from model.Model import Seq2Seq
from model.Encoder import Encoder
from model.Decoder import Decoder
from utilities.build_vocab import build_vocab

import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import math
import matplotlib.pyplot as plt


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            tok_types = batch.tok_types

            output, _ = model(src, trg[:, :-1], tok_types, SRC)

            # output = [batch size, trg len - 1, output dim]
            # trg = [batch size, trg len]

            output_dim = output.shape[-1]

            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg len - 1, output dim]
            # trg = [batch size * trg len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)



if __name__ == "__main__":
    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("------------------------------\n| Build vocab start ... | \n------------------------------")
    SRC, TRG, TOK_TYPES, BATCH_SIZE, train_iterator, valid_iterator, test_iterator, my_max_length = build_vocab(
        path_to_training_data='./dataset/dataset_final/',
        path_to_db_info='./dataset/database_information.csv'
    )
    print("------------------------------\n| Build vocab end ... | \n------------------------------")

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256  # it equals to embedding dimension # 原来256，可以改成standard的512试一试
    ENC_LAYERS = 3  # 3--> 6
    DEC_LAYERS = 3  # 3-->6
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1

    enc = Encoder(INPUT_DIM,
                  HID_DIM,
                  ENC_LAYERS,
                  ENC_HEADS,
                  ENC_PF_DIM,
                  ENC_DROPOUT,
                  device,
                  TOK_TYPES,
                  my_max_length
                  )

    dec = Decoder(OUTPUT_DIM,
                  HID_DIM,
                  DEC_LAYERS,
                  DEC_HEADS,
                  DEC_PF_DIM,
                  DEC_DROPOUT,
                  device,
                  my_max_length
                  )

    SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
    TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]

    model = Seq2Seq(enc, dec, SRC, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)  # define the Seq2Seq model

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    nl_acc = []
    nl_chart_acc = []


    model.load_state_dict(torch.load('./save_models/model_best.pt'))

    test_loss = evaluate(model, test_iterator, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    db_tables_columns = get_all_table_columns('./dataset/db_tables_columns.json')

    test_df = pd.read_csv('./dataset/dataset_final/test.csv')

    test_result = []  # tvBench_id, chart_type, hardness, ifChartTemplate, ifRight=1

    only_nl_cnt = 0
    only_nl_match = 0

    nl_template_cnt = 0
    nl_template_match = 0
    i = 0

    #     test_df = test_df[test_df['tvBench_id'] == '79']
    for index, row in tqdm(test_df.iterrows()):
        try:
            gold_query = row['labels'].lower()

            src = row['source'].lower()
            i += 1

            tok_types = row['token_types']

            #     translation,  attention = translate_sentence(
            #         src, SRC, TRG, TOK_TYPES, tok_types, model, device, my_max_length
            #     )

            translation, attention, enc_attention = translate_sentence_with_guidance(
                row['db_id'], gold_query.split(' ')[gold_query.split(' ').index('from') + 1],
                src, SRC, TRG, TOK_TYPES, tok_types, SRC, model, db_tables_columns, device, my_max_length
            )

            pred_query = ' '.join(translation).replace(' <eos>', '').lower()
            old_pred_query = pred_query

            if '[c]' not in src:
                # with template
                pred_query = postprocessing(gold_query, pred_query, True, src)

                nl_template_cnt += 1

                if ' '.join(gold_query.replace('"', "'").split()) == ' '.join(pred_query.replace('"', "'").split()):
                    nl_template_match += 1
                    test_result.append([
                        row['tvBench_id'],
                        row['chart'],
                        row['hardness'],
                        'chart_template',
                        1
                    ])
                else:
                    test_result.append([
                        row['tvBench_id'],
                        row['chart'],
                        row['hardness'],
                        'chart_template',
                        0
                    ])
            #             pass
            #                     print('---------with template---------------')
            #                     print(row['db_id'])
            #                     print(row['tvBench_id'])
            #                     print(src)
            #                     print(gold_query,'\n')
            #                     print(old_pred_query, '\n')
            #                     print(pred_query, '\n-------------------------\n')

            if '[c]' in src:
                # without template
                pred_query = postprocessing(gold_query, pred_query, False, src)

                only_nl_cnt += 1
                if ' '.join(gold_query.replace('"', "'").split()) == ' '.join(pred_query.replace('"', "'").split()):
                    only_nl_match += 1

                    test_result.append([
                        row['tvBench_id'],
                        row['chart'],
                        row['hardness'],
                        'only_nl',
                        1
                    ])
                else:
                    test_result.append([
                        row['tvBench_id'],
                        row['chart'],
                        row['hardness'],
                        'only_nl',
                        0
                    ])
        #             pass
        #                     print('---------without template---------------')
        #                     print(index)
        #                     print(row['db_id'])
        #                     print(row['tvBench_id'])
        #                     print(src)
        #                     print(gold_query,'\n')
        #                     print(old_pred_query, '\n')
        #                     print(pred_query, '\n-------------------------\n')

        except:
            print('error')

    nl_acc.append((only_nl_match) / only_nl_cnt)
    nl_chart_acc.append((nl_template_match) / nl_template_cnt)
    plt.plot(nl_acc)
    plt.plot(nl_chart_acc)
    plt.show()

    print('overall acc:', (only_nl_match + nl_template_match) / (only_nl_cnt + nl_template_cnt))
    print('only nl acc:', (only_nl_match) / only_nl_cnt)
    print('nl+template acc:', (nl_template_match) / nl_template_cnt)
