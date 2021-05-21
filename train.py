__author__ = "Yuyu Luo"

'''
This script handles the training process.
'''


import torch
import torch.nn as nn

from model.Model import Seq2Seq
from model.Encoder import Encoder
from model.Decoder import Decoder
from preprocessing.build_vocab import build_vocab

import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        tok_types = batch.tok_types

        optimizer.zero_grad()

        output, _ = model(src, trg[:, :-1], tok_types, SRC)

        # output = [batch size, trg len - 1, output dim]
        # trg = [batch size, trg len]

        output_dim = output.shape[-1]

        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg len - 1, output dim]
        # trg = [batch size * trg len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


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


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs



if __name__ == '__main__':

    SEED = 1234

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("------------------------------\n| Build vocab start ... | \n------------------------------")
    SRC, TRG, TOK_TYPES, BATCH_SIZE, train_iterator, valid_iterator, test_iterator, my_max_length =  build_vocab(
        path_to_training_data = './dataset/dataset_final/',
        path_to_db_info = './dataset/database_information.csv'
    )
    print("------------------------------\n| Build vocab end ... | \n------------------------------")

    INPUT_DIM = len(SRC.vocab)
    OUTPUT_DIM = len(TRG.vocab)
    HID_DIM = 256 # it equals to embedding dimension
    ENC_LAYERS = 3
    DEC_LAYERS = 3
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

    model = Seq2Seq(enc, dec, SRC, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device) # define the Seq2Seq model

    model.apply(initialize_weights)

    LEARNING_RATE = 0.0005

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

    print("------------------------------\n| Training start ... | \n------------------------------")


    N_EPOCHS = 100
    CLIP = 1

    train_loss_list, valid_loss_list = list(), list()

    best_valid_loss = float('inf')

    for epoch in range(N_EPOCHS):

        start_time = time.time()

        train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iterator, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # 保存最优的model
        if valid_loss < best_valid_loss:
            print('△○△○△○△○△○△○△○△○\nSave BEST model!\n△○△○△○△○△○△○△○△○△○')
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), './save_models/model_best.pt')

        # 保存每一个epoch的model
        print('△○△○△○△○△○△○△○△○\nSave model!\n△○△○△○△○△○△○△○△○△○')
        torch.save(model.state_dict(), './save_models/model_' + str(epoch + 1) + '.pt')

        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)

        print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')
        plt.plot(train_loss_list)
        plt.plot(valid_loss_list)
        plt.show()


