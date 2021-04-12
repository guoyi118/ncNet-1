__author__ = "Yuyu Luo"

import numpy as np
import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_visibility_matrix(SRC, each_src):
    each_src = np.array(each_src.to('cpu'))

    # find related index
    nl_beg_index = np.where(each_src == SRC.vocab['<nl>'])[0][0]
    nl_end_index = np.where(each_src == SRC.vocab['</nl>'])[0][0]
    template_beg_index = np.where(each_src == SRC.vocab['<template>'])[0][0]
    template_end_index = np.where(each_src == SRC.vocab['</template>'])[0][0]
    col_beg_index = np.where(each_src == SRC.vocab['<col>'])[0][0]
    col_end_index = np.where(each_src == SRC.vocab['</col>'])[0][0]
    value_beg_index = np.where(each_src == SRC.vocab['<value>'])[0][0]
    value_end_index = np.where(each_src == SRC.vocab['</value>'])[0][0]
    if SRC.vocab['[x]'] in each_src:
        x_index = np.where(each_src == SRC.vocab['[x]'])[0][0]
    else:
        # print('x')
        x_index = -1

    if SRC.vocab['[y]'] in each_src:
        y_index = np.where(each_src == SRC.vocab['[y]'])[0][0]
    else:
        # print('y')
        y_index = -1

    if SRC.vocab['[agg(y)]'] in each_src:
        agg_y_index = np.where(each_src == SRC.vocab['[agg(y)]'])[0][0]
    else:
        agg_y_index = -1
        # print('agg')

    if SRC.vocab['[xy]'] in each_src:
        xy_index = np.where(each_src == SRC.vocab['[xy]'])[0][0]
    else:
        xy_index = -1
        # print('xy')

    if SRC.vocab['[w]'] in each_src:
        where_index = np.where(each_src == SRC.vocab['[w]'])[0][0]
    else:
        where_index = -1
        # print('w')

    if SRC.vocab['[o]'] in each_src:
        other_index = np.where(each_src == SRC.vocab['[o]'])[0][0]
    else:
        other_index = -1
        # print('o')

    # init the visibility matrix
    v_matrix = np.zeros(each_src.shape * 2, dtype=int)

    # assign 1 to related cells

    # nl - (nl, template, col, value) self-attention
    v_matrix[nl_beg_index:nl_end_index, :] = 1
    v_matrix[:, nl_beg_index:nl_end_index] = 1

    # col-value self-attention
    v_matrix[col_beg_index:value_end_index, col_beg_index:value_end_index] = 1

    # template self-attention
    v_matrix[template_beg_index:template_end_index,
    template_beg_index:template_end_index] = 1

    # template - col/value self-attention
    # [x]/[y]/[agg(y)]/[o]/[w] <---> col
    # [w] <---> value
    # [c]/[o](order_type)/[i] <---> NL
    if x_index != -1:
        v_matrix[x_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, x_index] = 1
    if y_index != -1:
        v_matrix[y_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, y_index] = 1
    if agg_y_index != -1:
        v_matrix[agg_y_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, agg_y_index] = 1
    if other_index != -1:
        v_matrix[other_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, other_index] = 1
    if where_index != -1:
        v_matrix[where_index, col_beg_index:col_end_index] = 1
        v_matrix[where_index, value_beg_index:value_end_index] = 1

        v_matrix[col_beg_index:col_end_index, where_index] = 1
        v_matrix[value_beg_index:value_end_index, where_index] = 1
    if xy_index != -1:
        v_matrix[xy_index, col_beg_index:col_end_index] = 1
        v_matrix[col_beg_index:col_end_index, xy_index] = 1

    return v_matrix

class Seq2Seq(nn.Module):
    '''
    A transformer-based Seq2Seq model.
    '''
    def __init__(self,
                 encoder,
                 decoder,
                 SRC,
                 src_pad_idx,
                 trg_pad_idx,
                 device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    '''
    The source mask is created by checking where the source sequence is not equal to a <pad> token. 
    It is 1 where the token is not a <pad> token and 0 when it is. 
    It is then unsqueezed so it can be correctly broadcast when applying the mask to the energy, 
    which of shape [batch size, n heads, seq len, seq len].
    '''

    def make_visibility_matrix(self, src, SRC):
        '''
        building the visibility matrix here
        '''
        # src = [batch size, src len]
        batch_matrix = []
        for each_src in src:
            v_matrix = create_visibility_matrix(SRC, each_src)
            n_heads_matrix = [v_matrix] * 8 # 8 is the number of heads ...
            batch_matrix.append(np.array(n_heads_matrix))
        batch_matrix = np.array(batch_matrix)

        # batch_matrix = [batch size, n_heads, src_len, src_len]
        return torch.tensor(batch_matrix).to(device)

    def make_src_mask(self, src):
        # src = [batch size, src len]
        #         print(src)
        #         print(src.size())
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        #         print(src_mask)
        #         print(src_mask.size())
        #         print('==========\n')
        # src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # trg_pad_mask = [batch size, 1, 1, trg len]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()

        # trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        # trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg, tok_types, SRC):
        # src = [batch size, src len]
        # trg = [batch size, trg len]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        batch_visibility_matrix = self.make_visibility_matrix(src, SRC)  #######

        # src_mask = [batch size, 1, 1, src len]
        # trg_mask = [batch size, 1, trg len, trg len]

        enc_src, enc_attention = self.encoder(src, src_mask, tok_types, batch_visibility_matrix)  ######

        # enc_src = [batch size, src len, hid dim]

        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)

        # output = [batch size, trg len, output dim]
        # attention = [batch size, n heads, trg len, src len]

        return output, attention