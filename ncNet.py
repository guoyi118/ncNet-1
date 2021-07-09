__author__ = "Yuyu Luo"

import pandas as pd
import sqlite3
import re
import os

import torch
from model.VisAwareTranslation import translate_sentence_with_guidance, postprocessing
from model.Model import Seq2Seq
from model.Encoder import Encoder
from model.Decoder import Decoder
from preprocessing.build_vocab import build_vocab

from utilities.vis_rendering import VisRendering
from preprocessing.process_dataset import ProcessData4Training

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class ncNet(object):
    def __init__(self, trained_model):
        self.data = None
        self.db_id = ''
        self.table_id = ''
        self.db_tables_columns = None
        self.trained_model = trained_model # ./save_models/model_best.pt

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # TODO 能load已经build好的vocab吗？
        self.SRC, self.TRG, self.TOK_TYPES, BATCH_SIZE, train_iterator, valid_iterator, test_iterator, self.my_max_length = build_vocab(
            data_dir='./dataset/dataset_final/',
            db_info='./dataset/database_information.csv',
            batch_size=128,
            max_input_length=128
        )

        INPUT_DIM = len(self.SRC.vocab)
        OUTPUT_DIM = len(self.TRG.vocab)
        HID_DIM = 256  # it equals to embedding dimension
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
                      self.device,
                      self.TOK_TYPES,
                      self.my_max_length
                      )

        dec = Decoder(OUTPUT_DIM,
                      HID_DIM,
                      DEC_LAYERS,
                      DEC_HEADS,
                      DEC_PF_DIM,
                      DEC_DROPOUT,
                      self.device,
                      self.my_max_length
                      )

        SRC_PAD_IDX = self.SRC.vocab.stoi[self.SRC.pad_token]
        TRG_PAD_IDX = self.TRG.vocab.stoi[self.TRG.pad_token]

        self.ncNet = Seq2Seq(enc, dec, self.SRC, SRC_PAD_IDX, TRG_PAD_IDX, self.device).to(self.device)  # define the transformer-based ncNet
        self.ncNet.load_state_dict(torch.load(trained_model, map_location=self.device))


    def specify_dataset(self, data_type, db_url = None, table_name = None, data = None, data_url = None):
        '''
        :param data_type: sqlite3, csv, json
        :param db_url: db path for sqlite3 database, e.g., './dataset/database/flight/flight.sqlite'
        :param table_name: the table name in a sqlite3
        :param data: DataFrame for csv
        :param data_url: data path for csv or json
        :return: save the DataFrame in the self.data
        '''
        if data_type == 'csv':
            if data != None and data_url == None:
                self.data = data
            elif data == None and data_url != None:
                self.data = pd.read_csv(data_url)
            else:
                raise ValueError('Please only specify one of the data or data_url')
        elif data_type == 'json':
            if data == None and data_url != None:
                self.data = pd.read_json(data_url)
            else:
                raise ValueError('Read JSON from the json file, please only specify the "data_type" or "data_url"')

        elif data_type == 'sqlite3':
            # Create your connection.
            try:
                cnx = sqlite3.connect(db_url)
                self.data = pd.read_sql_query("SELECT * FROM " + table_name, cnx)
            except:
                raise ValueError('Errors in read table from sqlite3 database. \ndb_url: {0}\n table_name : {1} '.format(data_url, table_name))

        else:
            if data != None and type(data) == pd.core.frame.DataFrame:
                self.data = data
            else:
                raise ValueError('The data type must be one of the csv, json, sqlite3, or a DataFrame object.')

        self.db_id = 'temp_'+table_name
        self.table_id = table_name

        self.db_tables_columns = {
            self.db_id:{
                self.table_id: list(self.data.columns)
            }
        }

        if data_type == 'json' or data_type == 'sqlite3':
            # write to sqlite3 database
            if not os.path.exists('./dataset/database/'+self.db_id):
                os.makedirs('./dataset/database/'+self.db_id)

            conn = sqlite3.connect('./dataset/database/'+self.db_id+'/'+self.db_id+'.sqlite')

            self.data.to_sql(self.table_id, conn, if_exists='replace', index=False)

        self.DataProcesser = ProcessData4Training(db_url='./dataset/database')
        self.db_table_col_val_map = dict()
        table_cols = self.DataProcesser.get_table_columns(self.db_id)
        self.db_table_col_val_map[self.db_id] = dict()
        for table, cols in table_cols.items():
            col_val_map = self.DataProcesser.get_values_in_columns(self.db_id, table, cols, conditions='remove')
            self.db_table_col_val_map[self.db_id][table] = col_val_map

    def show_dataset(self):
        return self.data.head()


    def nl2vis(self, nl_question, chart_template=None, show_progress=None):
        # process and the nl_question and the chart template as input.
        # call the model to perform prediction
        # render the predicted query
        input_src, token_types = self.process_input(nl_question, chart_template)

        # print('=========================================================\n')
        # print('input_src:', input_src)
        # print('token_types:', token_types)

        pred_query, attention, enc_attention = translate_sentence_with_guidance(
            self.db_id, self.table_id, input_src, self.SRC, self.TRG, self.TOK_TYPES, token_types,
            self.SRC, self.ncNet, self.db_tables_columns, self.device, self.my_max_length, show_progress
        )

        pred_query = ' '.join(pred_query).replace(' <eos>', '').lower()
        if chart_template != None:
            pred_query = postprocessing(pred_query, pred_query, True, input_src)
        else:
            pred_query = postprocessing(pred_query, pred_query, False, input_src)
        # print('[NL Question]:', nl_question)
        # print('[Chart Template]:', chart_template)
        # print('[Predicted VIS Query]:', ' '.join(pred_query.replace('"', "'").split()))

        create_vis = VisRendering()

        vis_query = create_vis.parse_output_query(
            './dataset/database/',
            self.db_id,
            self.table_id,
            ' '.join(pred_query.replace('"', "'").split())
        )

        data4vis = create_vis.query_sqlite3(
            './dataset/database/',
            self.db_id,
            vis_query['data_part']['sql_part']
        )
        print('The Predicted VIS Query:')
        print(pred_query)
        print('The Predicted VIS Result:')
        create_vis.render_vis(data4vis, vis_query)
        print('\n')



    def process_input(self, nl_question, chart_template):

        def get_token_types(input_source):

            token_types = ''

            for ele in re.findall('<nl>.*</nl>', input_source)[0].split(' '):
                token_types += ' nl'

            for ele in re.findall('<template>.*</template>', input_source)[0].split(' '):
                token_types += ' template'

            for ele in re.findall('<col>.*</col>', input_source)[0].split(' '):
                token_types += ' col'

            for ele in re.findall('<value>.*</value>', input_source)[0].split(' '):
                token_types += ' value'

            token_types = token_types.strip()

            return token_types

        def fix_chart_template(chart_template = None):
            query_template = 'visualize [c] select [x], [agg(y)] from [d] where [w] group by [xy] bin [x] by [i] order by [xy] [t]'

            if chart_template != None:
                try:
                    query_template = query_template.replace('[c]', chart_template['chart'])
                except:
                    raise ValueError('Error at settings of chart type!')

                try:
                    if 'sorting_options' in chart_template and chart_template['sorting_options'] != None:
                        if 'axis' in chart_template['sorting_options']:
                            if chart_template['sorting_options']['axis'].lower() == 'x':
                                query_template = query_template.replace('order by [xy]', 'order by [x]')
                            elif chart_template['sorting_options']['axis'].lower() == 'y':
                                query_template = query_template.replace('order by [xy]', 'order by [y]')
                            else:
                                query_template = query_template.replace('order by [xy]', 'order by [o]')

                        if 'type' in chart_template['sorting_options']:
                            if chart_template['sorting_options']['type'].lower() == 'desc':
                                query_template = query_template.replace('[t]', 'desc')
                            elif chart_template['sorting_options']['type'].lower() == 'asc':
                                query_template = query_template.replace('[t]', 'asc')
                            else:
                                raise ValueError('Unknown order by settings, the order-type must be "desc", or "asc"')
                except:
                    raise ValueError('Error at settings of sorting!')

                return query_template
            else:
                return query_template

        query_template = fix_chart_template(chart_template)
        # get a list of mentioned values in the NL question
        col_names, value_names = self.DataProcesser.get_mentioned_values_in_NL_question(
            self.db_id, self.table_id, nl_question, db_table_col_val_map=self.db_table_col_val_map
        )
        col_names = ' '.join(str(e) for e in col_names)
        # col_names = self.table_id + ' ' + col_names
        value_names = ' '.join(str(e) for e in value_names)
        input_src = "<nl> {} </nl> <template> {} </template> <col> {} </col> <value> {} </value>".format(nl_question, query_template,
                                                                                                 col_names,
                                                                                                 value_names).lower()
        token_types = get_token_types(input_src)

        return input_src, token_types



if __name__ == '__main__':
    ncNet = ncNet(
        trained_model='./save_models/model_best.pt'
    )
    ncNet.specify_dataset(
        data_type='sqlite3',
        db_url='./dataset/database/covid19/covid19.sqlite',
        table_name='us_states'
    )
    ncNet.nl2vis(
        nl_question='I want to know the proportion of total number by each case in Massachusetts on 2021-02-27',
        chart_template=None
    )
