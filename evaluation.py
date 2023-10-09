import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def R2(y_test, y_pred):
    return 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - y_pred.reshape(-1, 1)) ** 2)) / (
        sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))

def ls_analysis(df: pd.DataFrame):
    long = df.groupby(df.index).apply(lambda x: x[x['y_pred'] >= x['y_pred'].quantile(0.9)].loc[:, ['y_true']].mean())
    short = df.groupby(df.index).apply(lambda x: x[x['y_pred'] <= x['y_pred'].quantile(0.1)].loc[:, ['y_true']].mean())

    ls_ret = long - short
    return long, short, ls_ret

if __name__ == '__main__':
    model = 'Ridge'
    way = 'pca_ridge_test'
    path = '/20230328代码与数据/美国市场预测结果'

    df = pd.read_csv("/20230328代码与数据/美国市场数据/final_data——1965-2020.csv", encoding='utf-8').rename(columns={'R-1': 'y_true'})
    df['ym'] = df['DATE'].astype(str).apply(lambda x: x[0:4] + '-' + x[4:6])
    df = df.set_index(df['ym'])
    df.index = pd.to_datetime(df.index)

    y_df = pd.DataFrame(df['2001': '2019'].loc[:, ['y_true']])
    pre_list = []
    for year in range(1, 20):
        pre_list.append(np.array(pd.read_csv(f'{path}/{way}/{model}-{year}.csv')).reshape(-1, 1))
    y_df['y_pred'] = np.vstack(pre_list).reshape(-1, )

    y_df = y_df.reset_index()
    y_df.sort_values('ym', inplace=True)
    y_df = y_df.set_index('ym')
    y_df['y_pred'] = y_df.groupby(y_df.index).apply(lambda x: x['y_pred'] * x['y_true'].std() + x['y_true'].mean()).values
    print('R2:', R2(y_test=y_df['y_true'].values, y_pred=y_df['y_pred'].values))
    long, short, ls_ret = ls_analysis(y_df)
    print(long.mean(), short.mean(), (long-short).mean(), (long-short).mean()/(long-short).std())
    plt.plot((long-short).cumsum())
    plt.show()