__author__ = "Yuyu Luo"

import torch
from torchtext.data import Field, TabularDataset, BucketIterator

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_vocab(data_dir, db_info, batch_size, max_input_length):

    def tokenizer(text):
        return text.split(' ')

    # def tokenizer_src(text):
    #     return text.split(' ')

    SRC = Field(tokenize=tokenizer,  # 指定分词函数
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    TRG = Field(tokenize=tokenizer,  # Visualization Query应该有特有的分词函数
                init_token='<sos>',
                eos_token='<eos>',
                lower=True,
                batch_first=True)

    TOK_TYPES = Field(tokenize=tokenizer,  # 指定分词函数
                      init_token='<sos>',
                      eos_token='<eos>',
                      lower=True,
                      batch_first=True)

    train_data, valid_data, test_data = TabularDataset.splits(
        path=data_dir, format='csv', skip_header=True,
        train='train.csv', validation='dev.csv', test='test.csv',
        fields=[
            ('tvBench_id', None),
            ('db_id', None),
            ('chart', None),
            ('hardness', None),
            ('query', None),
            ('question', None),
            ('mentioned_columns', None),
            ('mentioned_values', None),
            ('query_template', None),
            ('src', SRC),
            ('trg', SRC),
            ('tok_types', TOK_TYPES)
        ])

    DB_TOK = Field(
        tokenize=tokenizer,  # 指定分词函数
        lower=True,
        batch_first=True
    )

    db_information = TabularDataset(
        path=db_info,
        format='csv',
        skip_header=True,
        fields=[
            ('table', SRC),
            ('column', SRC),
            ('value', SRC)
        ]
    )

    # 加载数据后可以建立词典，建立词典的时候可以使用预训练的word vector
    # TEXT.build_vocab(train, vectors="glove.6B.100d") #example
    '''
    TODO: 更新字典集大小
    '''
    # SRC.build_vocab(train_data, valid_data, test_data, min_freq = 1, vectors="glove.6B.100d")
    # TRG.build_vocab(train_data, valid_data, test_data, min_freq = 1, vectors="glove.6B.100d")
    # TOK_TYPES.build_vocab(train_data, valid_data, test_data, min_freq = 1, vectors="glove.6B.100d")

    # SRC.build_vocab(train_data, valid_data, test_data, db_information, min_freq = 2)
    # TRG.build_vocab(train_data, valid_data, test_data, db_information, min_freq = 2)
    # TOK_TYPES.build_vocab(train_data, valid_data, test_data, db_information, min_freq = 2)

    SRC.build_vocab(train_data, valid_data, test_data, db_information, min_freq=2)
    # TRG.build_vocab(train_data, valid_data, test_data, db_information, min_freq = 2,  vectors="glove.6B.100d")
    TRG = SRC
    TOK_TYPES.build_vocab(train_data, valid_data, test_data, db_information, min_freq=2)


    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data), sort=False,
        batch_size=batch_size,
        device=device)

    return SRC, TRG, TOK_TYPES, batch_size, train_iterator, valid_iterator, test_iterator, max_input_length