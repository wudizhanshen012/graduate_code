# -*- coding: UTF-8 -*-

import os
import pandas as pd

data_path = '/share/home/qshz79/share/home/qshz79/graduate_code/graduate_code' + '/us_code/data/'
how_to_feature_extraction = 'pca'   # 可以选择origin(不处理)/pca/pls/ae
train_model_list = ['OLS', 'LASSO', 'RIDGE', 'ENET', 'RF', 'NN3']  #
start_year = 2001
end_year = 2019

base_col = ['permno', 'ym', 'sic2']
y_col = ['ret+1']
x_col = ['mvel1', 'beta', 'betasq', 'chmom', 'dolvol', 'idiovol', 'indmom', 'mom1m', 'mom6m', 'mom12m', 'mom36m', 'pricedelay',
         'turn', 'absacc', 'acc', 'age', 'agr', 'cashdebt', 'cashpr', 'cfp', 'cfp_ia', 'chatoia', 'chcsho', 'chempia', 'chinv',
         'chpmia', 'convind', 'currat', 'depr', 'divi', 'divo', 'dy', 'egr', 'ep', 'gma', 'grcapx', 'grltnoa', 'herf', 'hire',
         'invest', 'lev', 'lgr', 'mve_ia', 'operprof', 'orgcap', 'pchcapx_ia', 'pchcurrat', 'pchdepr', 'pchgm_pchsale', 'pchquick',
         'pchsale_pchinvt', 'pchsale_pchrect', 'pchsale_pchxsga', 'pchsaleinv', 'pctacc', 'ps', 'quick', 'rd', 'rd_mve', 'rd_sale',
         'realestate', 'roic', 'salecash', 'saleinv', 'salerec', 'secured', 'securedind', 'sgr', 'sin', 'sp', 'tang', 'tb', 'aeavol',
         'cash', 'chtx', 'cinvest', 'ear', 'nincr', 'roaq', 'roavol', 'roeq', 'rsup', 'stdacc', 'stdcf', 'ms', 'baspread', 'ill',
         'maxret', 'retvol', 'std_dolvol', 'std_turn', 'zerotrade', 'bm', 'bm_ia']

macro_col = ['Dp', 'Ep', 'Bm', 'Ntis', 'Tbl', 'Tms', 'Dfy', 'Svar']
category_factor_dict = {
    'size': ['mvel1', 'mve_ia', 'herf', 'chinv', 'chcsho', 'age'],
    'beta': ['beta', 'betasq'],
    'mom': ['mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom', 'maxret', 'indmom'],
    'liq': ['std_dolvol', 'zerotrade', 'std_turn', 'ill', 'turn', 'dolvol', 'pricedelay', 'chatoia', 'salerec'],
    'vol': ['retvol', 'aeavol', 'idiovol', 'ear', 'roavol'],
    'bpr': ['bm', 'bm_ia', 'cfp', 'cfp_ia', 'sp', 'invest', 'realestate', 'depr', 'cashpr'],
    'ey': ['roeq', 'ep', 'roaq', 'divo', 'absacc', 'divi', 'chempia', 'nincr', 'chpmia', 'stdacc', 'chtx', 'cash', 'roic',
      'stdcf', 'dy', 'acc', 'pctacc', 'saleinv', 'operprof', 'pchsale_pchrect', 'salecash', 'tb', 'gma', 'pchdepr'],
    'growth': ['grltnoa', 'egr', 'orgcap', 'sgr', 'pchgm_pchsale', 'rsup', 'pchsaleinv', 'rd_sale', 'rd_mve', 'rd',
          'cinvest', 'pchsale_pchxsga', 'pchsale_pchinvt', 'agr', 'grcapx', 'hire'],
    'lever': ['securedind', 'secured', 'lev', 'pchquick', 'pchcapx_ia', 'lgr', 'quick', 'ps', 'tang', 'currat', 'ms',
         'pchcurrat', 'cashdebt', 'convind']
}

data = pd.read_pickle(data_path + 'processed_data/origin_data.pkl')


for category in category_factor_dict:
    temp = pd.read_csv(data_path + f'/processed_data/ae_{category}.csv')
    data['ae_' + category] = temp['ae_' + category]
    print(category, 'done!')

AE_data = data[['permno', 'ym', 'sic2', 'ret+1'] + ['sin', 'baspread'] + list(map(lambda s: 'ae_' + s, list(category_factor_dict.keys()))) + macro_col]
AE_data.to_pickle(data_path + 'processed_data/ae_data.pkl')