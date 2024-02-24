import numpy as np
import pandas as pd
import os
from us_code.tool import cal_R2
from us_code.config import data_path, y_col

feature_extraction_way = 'pls'
model = 'LASSO'

df_list = []
for year in range(2001, 2020):
    prediction_data_path = os.path.join(data_path, f'prediction/{feature_extraction_way}/{model}-{year}.csv')
    a = pd.read_csv(prediction_data_path)
    df_list.append(a)

df = pd.concat(df_list)

print('R2:', cal_R2(y_test=df['ret+1'], pred_y=df['y_pred']))

long_portfolio = df.groupby('ym').apply(lambda x: x[x['y_pred'] >= x['y_pred'].quantile(0.9)])
short_portfolio = df.groupby('ym').apply(lambda x: x[x['y_pred'] <= x['y_pred'].quantile(0.1)])
long_portfolio = long_portfolio.reset_index(drop=True)
short_portfolio = short_portfolio.reset_index(drop=True)
long_ret = long_portfolio.groupby('ym')[y_col[0]].mean().rename('long')
short_ret = short_portfolio.groupby('ym')[y_col[0]].mean().rename('short')
all_ret = pd.concat([long_ret, short_ret], axis=1)
all_ret['ls_ret'] = all_ret['long'] - all_ret['short']
annual_ret = all_ret['ls_ret'].mean() * 12
annual_std = all_ret['ls_ret'].std() * np.sqrt(12)
sharpe_ratio = annual_ret / annual_std
print(f'年化收益率:{annual_ret}, 夏普比率: {sharpe_ratio}')