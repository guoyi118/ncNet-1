'''
This file is for convert vis query to the Vega-Lite object
'''

# from IPython.core.interactiveshell import InteractiveShell
# InteractiveShell.ast_node_interactivity = 'all'

# from vega import VegaLite

from IPython.display import display
def Vega(spec):
    bundle = {}
    bundle['application/vnd.vega.v5+json'] = spec
    display(bundle, raw=True)

def VegaLite(spec):
    bundle = {}
    bundle['application/vnd.vegalite.v4+json'] = spec
    display(bundle, raw=True)

import pandas as pd
import sqlite3
import json

# TODO: 参考NL4DV，return的object list里面需要有能直接渲染成可视化结果的，也有便被Python解析的object、

class VisRendering(object):
    def __init__(self):
        self.VegaLiteSpec = {
            'Bar': {
                "data": {"values": []},
                "mark": "bar",
                "encoding": {
                    "x": {"field": "x", "type": "nominal", "sort": "x"},
                    "y": {"field": "y", "type": "quantitative"}
                }
            },
            'Stacked Bar': {
                "data": {"values": []},
                "mark": "bar",
                "encoding": {
                    "x": {"field": "x", "type": "nominal", "sort": "x"},
                    "y": {"field": "y", "type": "quantitative"}
                }
            },
            'Pie': {
                "data": {"values": []},
                "mark": "arc",
                "encoding": {
                    "color": {"field": "x", "type": "nominal"},
                    "theta": {"field": "y", "type": "quantitative"}
                }
            },
            'Line': {
                "data": {"values": []},
                "mark": "line",
                "encoding": {
                    "x": {"field": "x", "type": "nominal", "sort": "x"},
                    "y": {"field": "y", "type": "quantitative"}
                }
            },
            'Grouping Line': {
                "data": {"values": []},
                "mark": "bar",
                "encoding": {
                    "x": {"field": "x", "type": "nominal", "sort": "x"},
                    "y": {"field": "y", "type": "quantitative"}
                }
            },
            'Scatter': {
                "data": {"values": []},
                "mark": "point",
                "encoding": {
                    "x": {"field": "x", "type": "nominal", "sort": "x"},
                    "y": {"field": "y", "type": "quantitative"}
                }
            },
            'Grouping Scatter': {
                "data": {"values": []},
                "mark": "point",
                "encoding": {
                    "x": {"field": "x", "type": "nominal", "sort": "x"},
                    "y": {"field": "y", "type": "quantitative"}
                }
            },
        }

    def inf_chart_type_from_VQL(self, VQL):

        VQL = VQL.lower()

        V_keywords = VQL.split(' ')
        base_chart = V_keywords[V_keywords.index('visualize') + 1]
        # if two grouping.
        if 'group' in V_keywords and V_keywords.index('group') + 3 < len(V_keywords) and V_keywords[
            V_keywords.index('group') + 3] == ',':
            if base_chart == 'bar':
                return "Stacked Bar"
            elif base_chart == 'line':
                return 'Grouping Line'
            elif base_chart == 'scatter':
                return 'Grouping Scatter'
            else:
                print('xx')
        # if grouping + binning
        elif 'group by' in VQL and 'bin' in VQL:
            if base_chart == 'bar':
                return "Stacked Bar"
            elif base_chart == 'line':
                return 'Grouping Line'
            elif base_chart == 'scatter':
                return 'Grouping Scatter'
            else:
                print('yy')
        # if grouping for scatter
        elif base_chart == 'scatter' and 'group by' in VQL:
            return 'Grouping Scatter'

        # TODO binning?

        # grouping for line,  select x, y ... grouping by z
        elif base_chart == 'line' and 'group by' in VQL:
            if V_keywords[V_keywords.index('select') + 1] != V_keywords[V_keywords.index('group') + 2]:
                return 'Grouping Line'
            else:
                return base_chart.title()
        else:
            return base_chart.title()

    def query_sqlite3(self, path_to_db, db_id, sql_query):
        '''
        take a sql query as input and return the query result in JSON format
        '''
        # Create your connection.
        con = sqlite3.connect(path_to_db + '/' + db_id + '/' + db_id + '.sqlite')

        '''
        for cases insensititve comparison in WHERE clause , COLLATE NOCASE
        '''
        sql_query = sql_query.lower()
        sql_query_list = sql_query.split(' ')
        if 'where' in sql_query_list:
            target_keywords = ['group', 'bin', 'order']
            target_key = ''
            for each in sql_query_list[sql_query_list.index('where'):]:
                if each in target_keywords:
                    target_key = each
                    break
            if target_key == '':
                sql_query += ' COLLATE NOCASE'
            else:
                sql_query_list.insert(sql_query_list.index(target_key), 'COLLATE NOCASE')
                sql_query = ' '.join(sql_query_list)

        df = pd.read_sql_query(sql_query, con)

        df.columns = map(str.lower, df.columns)  # all column names to lower cases

        df_json = df.to_json(orient='records')
        return json.loads(df_json)

    def parse_output_query(self, path_to_db, db_id, table_id, query):
        # TODO how to handle binning operation?
        '''
        take the output query (format as nvBench-SIGMOD 2021) as input, and output:
        vis_query = {
            'vis_part': ...
            'data_part':{
                'sql_part': ... ,
                'binning_part': ... ,
            },
            'VQL': '...',
            'vage_mapping':{
                'x': '',
                'y': '',
                'color': '',
                'binning': '',
                'sort': '-x'/'x'/'y'/'-y'
            }
        }
        '''

        query = query.lower()

        vis_query = {
            'vis_part': '',
            'data_part': {
                'sql_part': '',
                'binning': ''
            },
            'VQL': query,
            'vega_mapping': {
                'x': '',
                'y': '',
                'color': '',
                'sort': '',
            }
        }

        query_list = query.split(' ')
        if 'bin' in query_list:
            vis_query['data_part']['binning'] = ' '.join(query_list[query_list.index('bin'):])
            vis_query['data_part']['sql_part'] = ' '.join(
                query_list[query_list.index('select'):query_list.index('bin')])
        else:
            vis_query['data_part']['sql_part'] = ' '.join(query_list[query_list.index('select'):])

        vis_query['vis_part'] = self.inf_chart_type_from_VQL(query)

        axis = query_list[query_list.index('select') + 1:query_list.index('from')]
        axis[:] = [x for x in axis if x != ',']

        # get the column types
        try:
            con = sqlite3.connect(path_to_db + '/' + db_id + '/' + db_id + '.sqlite')
            name_type_pairs = pd.read_sql_query('PRAGMA TABLE_INFO('+table_id+')', con)
            col_names_types = dict(zip(name_type_pairs['name'], name_type_pairs['type']))
            agg_func = ['sum', 'avg', 'count', 'cnt', 'max', 'min']
            agg_col_name_types = dict()
            for k, v in col_names_types.items():
                for agg in agg_func:
                    agg_col_name_types[agg+'('+k+')'] = v

            def Merge(dict1, dict2):
                res = {**dict1, **dict2}
                return res
            col_names_types = Merge(col_names_types, agg_col_name_types)

            if len(axis) == 2:
                if col_names_types[axis[0]] == 'REAL' or col_names_types[axis[0]] == 'INTEGER':
                    vis_query['vega_mapping']['y'] = axis[0]
                    vis_query['vega_mapping']['x'] = axis[1]
                else:
                    vis_query['vega_mapping']['x'] = axis[0]
                    vis_query['vega_mapping']['y'] = axis[1]
            elif len(axis) == 3:
                number_cnt = [0,0,0]
                for i in range(len(axis)):
                    if col_names_types[axis[i]] in ['REAL', 'INTEGER']:
                        number_cnt[i] = 1
                if sum(number_cnt) == 2: # 2 numbers + 1 category
                    vis_query['vega_mapping']['color'] = axis[number_cnt.index(0)]
                    if number_cnt.index(0) == 0:
                        vis_query['vega_mapping']['x'] = axis[1]
                        vis_query['vega_mapping']['y'] = axis[2]
                    elif number_cnt.index(0) == 1:
                        vis_query['vega_mapping']['x'] = axis[0]
                        vis_query['vega_mapping']['y'] = axis[2]
                    else:
                        vis_query['vega_mapping']['x'] = axis[0]
                        vis_query['vega_mapping']['y'] = axis[1]
                elif sum(number_cnt) == 1: # 1 number + 2 categories
                    vis_query['vega_mapping']['y'] = axis[number_cnt.index(1)]
                    if number_cnt.index(1) == 0:
                        vis_query['vega_mapping']['x'] = axis[1]
                        vis_query['vega_mapping']['color'] = axis[2]
                    elif number_cnt.index(1) == 1:
                        vis_query['vega_mapping']['x'] = axis[0]
                        vis_query['vega_mapping']['color'] = axis[2]
                    else:
                        vis_query['vega_mapping']['x'] = axis[0]
                        vis_query['vega_mapping']['color'] = axis[1]
                else:
                    vis_query['vega_mapping']['x'] = axis[0]
                    vis_query['vega_mapping']['y'] = axis[1]
                    vis_query['vega_mapping']['color'] = axis[2]
            else:
                print('unexpected the number of axies: ', len(axis))
        except:

            if len(axis) == 2:
                vis_query['vega_mapping']['x'] = axis[0]
                vis_query['vega_mapping']['y'] = axis[1]
            elif len(axis) == 3:
                vis_query['vega_mapping']['x'] = axis[0]
                vis_query['vega_mapping']['y'] = axis[1]
                vis_query['vega_mapping']['color'] = axis[2]
            else:
                print('unexpected the number of axies: ', len(axis))

        if 'order' in query_list and query_list[query_list.index('order') + 1] == 'by':
            sort_axis = query_list[query_list.index('order') + 2]
            sort_order = query_list[query_list.index('order') + 3]
            if sort_axis == vis_query['vega_mapping']['x'] and sort_order == 'desc':
                vis_query['vega_mapping']['sort'] = '-x'
            if sort_axis == vis_query['vega_mapping']['x'] and sort_order == 'asc':
                vis_query['vega_mapping']['sort'] = 'x'
            if sort_axis == vis_query['vega_mapping']['y'] and sort_order == 'desc':
                vis_query['vega_mapping']['sort'] = '-y'
            if sort_axis == vis_query['vega_mapping']['y'] and sort_order == 'asc':
                vis_query['vega_mapping']['sort'] = 'y'

        return vis_query

    def render_vis(self, data4vis, vis_query):
        self.VegaLiteSpec[vis_query['vis_part']]['data']['values'] = data4vis

        if vis_query['vis_part'] == 'Pie':
            self.VegaLiteSpec[vis_query['vis_part']]['encoding']['color']['field'] = vis_query['vega_mapping']['x']
            self.VegaLiteSpec[vis_query['vis_part']]['encoding']['theta']['field'] = vis_query['vega_mapping']['y']
        else:
            self.VegaLiteSpec[vis_query['vis_part']]['encoding']['x']['field'] = vis_query['vega_mapping']['x']
            self.VegaLiteSpec[vis_query['vis_part']]['encoding']['y']['field'] = vis_query['vega_mapping']['y']
            if vis_query['vega_mapping']['color'] != '':
                self.VegaLiteSpec[vis_query['vis_part']]['encoding']['color'] = {
                    'field': vis_query['vega_mapping']['color'].lower(),
                    "type": "nominal"
                }
            if vis_query['vega_mapping']['sort'] != '':
                self.VegaLiteSpec[vis_query['vis_part']]['encoding']['x']['sort'] = vis_query['vega_mapping']['sort']

        return VegaLite(self.VegaLiteSpec[vis_query['vis_part']])
