from us_code.config import *
from us_code.tool import cal_R2
from us_code.model import *
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import PredefinedSplit
from tensorflow.keras.layers import Input

print(how_to_feature_extraction, 'start')
# read_data
data = pd.read_pickle(data_path + f'processed_data/{how_to_feature_extraction}_data.pkl')

# TODO origin的data columns没弄好，所以需要单独处理，后续可以考虑把这个问题解决掉，通过在1_process_data.py中调整最后保存的data的columns
if how_to_feature_extraction != 'origin':
    x_col = list(data.iloc[:, 4:-len(macro_col)].columns)

# OLS
if 'OLS' in train_model_list:
    loss, Y_test, Y_pre = [], [], []
    weights = []
    for year in range(start_year, end_year + 1):
        x_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, x_col + macro_col]
        y_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, y_col[0]]

        x_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:,
                x_col + macro_col]
        y_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        x_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                 x_col + macro_col]
        y_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:, y_col[0]]

        x_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, x_col + macro_col]
        y_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        model = linear_regression()
        history = model.fit(x_train_val, y_train_val,
                            )

        pred_y = model.predict(x_test).reshape(len(y_test), 1)
        print('预测值为:', pred_y)

        R2 = cal_R2(y_test=y_test, pred_y=pred_y)
        print(f'{year}年OLS的R方：', R2)

        saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                   ['permno', 'ym', 'sic2', y_col[0]]]
        saved_df['y_pred'] = pred_y

        if not os.path.exists(data_path + f'prediction/{how_to_feature_extraction}'):
            os.mkdir(data_path + f'prediction/{how_to_feature_extraction}')
        saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/OLS-{year}.csv', index=False)

        # 因子重要性
        if how_to_feature_extraction == 'origin':
            pass
        else:
            for x in x_col:
                zero_x_test = x_test.copy()
                zero_x_test[x] = 0
                zero_pred_y = model.predict(zero_x_test).reshape(len(y_test), 1)
                saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                           ['permno', 'ym', 'sic2', y_col[0]]]
                saved_df['y_pred'] = zero_pred_y
                saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/OLS-{year}-{x}.csv', index=False)

        loss.append(history)
        Y_test.append(y_test)
        Y_pre.append(pred_y)
        weights.append(list(model.coef_))
        print('第{}年完成'.format(year))

    analysis_data = data[(data['ym'] >= f'{start_year}-01-01') & (data['ym'] <= f'{end_year}-12-31')].loc[:,
                    ['ym', y_col[0]]]
    pre = np.vstack(Y_pre)
    analysis_data['pred'] = pre.reshape(len(pre), )

    R2 = cal_R2(y_test=analysis_data[y_col[0]], pred_y=analysis_data['pred'])
    print('最终OLS的R方：', R2)

    long_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] >= x['pred'].quantile(0.9)])
    short_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] <= x['pred'].quantile(0.1)])
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
else:
    print('OLS不在train_model_list中，跳过训练')

# LASSO
if 'LASSO' in train_model_list:
    loss, Y_test, Y_pre = [], [], []
    weights = []
    for year in range(start_year, end_year + 1):
        x_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, x_col + macro_col]
        y_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, y_col[0]]

        x_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:,
                x_col + macro_col]
        y_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        x_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                 x_col + macro_col]
        y_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:, y_col[0]]

        x_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, x_col + macro_col]
        y_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
        test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
        ps = PredefinedSplit(test_fold=test_fold)

        model = lasso(ps=ps)
        history = model.fit(x_train_val, y_train_val)

        pred_y = model.predict(x_test).reshape(len(y_test), 1)
        print('预测值为:', pred_y)

        R2 = cal_R2(y_test=y_test, pred_y=pred_y)
        print(f'{year}年LASSO的R方：', R2)

        saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                   ['permno', 'ym', 'sic2', y_col[0]]]
        saved_df['y_pred'] = pred_y

        if not os.path.exists(data_path + f'prediction/{how_to_feature_extraction}'):
            os.mkdir(data_path + f'prediction/{how_to_feature_extraction}')
        saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/LASSO-{year}.csv', index=False)

        # 因子重要性
        if how_to_feature_extraction == 'origin':
            pass
        else:
            for x in x_col:
                zero_x_test = x_test.copy()
                zero_x_test[x] = 0
                zero_pred_y = model.predict(zero_x_test).reshape(len(y_test), 1)
                saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                           ['permno', 'ym', 'sic2', y_col[0]]]
                saved_df['y_pred'] = zero_pred_y
                saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/LASSO-{year}-{x}.csv', index=False)

        loss.append(history)
        Y_test.append(y_test)
        Y_pre.append(pred_y)
        weights.append(list(model.coef_))
        print('第{}年完成'.format(year))

    analysis_data = data[(data['ym'] >= f'{start_year}-01-01') & (data['ym'] <= f'{end_year}-12-31')].loc[:,
                    ['ym', y_col[0]]]
    pre = np.vstack(Y_pre)
    analysis_data['pred'] = pre.reshape(len(pre), )

    R2 = cal_R2(y_test=analysis_data[y_col[0]], pred_y=analysis_data['pred'])
    print('最终LASSO的R方：', R2)

    long_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] >= x['pred'].quantile(0.9)])
    short_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] <= x['pred'].quantile(0.1)])
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
else:
    print('LASSO不在train_model_list中，跳过训练')

# RIDGE
if 'RIDGE' in train_model_list:
    loss, Y_test, Y_pre = [], [], []
    weights = []
    for year in range(start_year, end_year + 1):
        x_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, x_col + macro_col]
        y_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, y_col[0]]

        x_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:,
                x_col + macro_col]
        y_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        x_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                 x_col + macro_col]
        y_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:, y_col[0]]

        x_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, x_col + macro_col]
        y_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
        test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
        ps = PredefinedSplit(test_fold=test_fold)

        model = ridge(ps=ps)
        history = model.fit(x_train_val, y_train_val)

        pred_y = model.predict(x_test).reshape(len(y_test), 1)
        print('预测值为:', pred_y)

        R2 = cal_R2(y_test=y_test, pred_y=pred_y)
        print(f'{year}年RIDGE的R方：', R2)

        saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                   ['permno', 'ym', 'sic2', y_col[0]]]
        saved_df['y_pred'] = pred_y

        if not os.path.exists(data_path + f'prediction/{how_to_feature_extraction}'):
            os.mkdir(data_path + f'prediction/{how_to_feature_extraction}')
        saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/RIDGE-{year}.csv', index=False)

        # 因子重要性
        if how_to_feature_extraction == 'origin':
            pass
        else:
            for x in x_col:
                zero_x_test = x_test.copy()
                zero_x_test[x] = 0
                zero_pred_y = model.predict(zero_x_test).reshape(len(y_test), 1)
                saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                           ['permno', 'ym', 'sic2', y_col[0]]]
                saved_df['y_pred'] = zero_pred_y
                saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/RIDGE-{year}-{x}.csv', index=False)

        loss.append(history)
        Y_test.append(y_test)
        Y_pre.append(pred_y)
        weights.append(list(model.coef_))
        print('第{}年完成'.format(year))

    analysis_data = data[(data['ym'] >= f'{start_year}-01-01') & (data['ym'] <= f'{end_year}-12-31')].loc[:,
                    ['ym', y_col[0]]]
    pre = np.vstack(Y_pre)
    analysis_data['pred'] = pre.reshape(len(pre), )

    R2 = cal_R2(y_test=analysis_data[y_col[0]], pred_y=analysis_data['pred'])
    print('最终RIDGE的R方：', R2)

    long_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] >= x['pred'].quantile(0.9)])
    short_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] <= x['pred'].quantile(0.1)])
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
else:
    print('RIDGE不在train_model_list中，跳过训练')

# ENET
if 'ENET' in train_model_list:
    loss, Y_test, Y_pre = [], [], []
    weights = []
    for year in range(start_year, end_year + 1):
        x_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, x_col + macro_col]
        y_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, y_col[0]]

        x_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:,
                x_col + macro_col]
        y_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        x_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                 x_col + macro_col]
        y_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:, y_col[0]]

        x_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, x_col + macro_col]
        y_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
        test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
        ps = PredefinedSplit(test_fold=test_fold)

        model = ridge(ps=ps)
        history = model.fit(x_train_val, y_train_val)

        pred_y = model.predict(x_test).reshape(len(y_test), 1)
        print('预测值为:', pred_y)

        R2 = cal_R2(y_test=y_test, pred_y=pred_y)
        print(f'{year}年ENET的R方：', R2)

        saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                   ['permno', 'ym', 'sic2', y_col[0]]]
        saved_df['y_pred'] = pred_y

        if not os.path.exists(data_path + f'prediction/{how_to_feature_extraction}'):
            os.mkdir(data_path + f'prediction/{how_to_feature_extraction}')
        saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/ENET-{year}.csv', index=False)

        # 因子重要性
        if how_to_feature_extraction == 'origin':
            pass
        else:
            for x in x_col:
                zero_x_test = x_test.copy()
                zero_x_test[x] = 0
                zero_pred_y = model.predict(zero_x_test).reshape(len(y_test), 1)
                saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                           ['permno', 'ym', 'sic2', y_col[0]]]
                saved_df['y_pred'] = zero_pred_y
                saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/ENET-{year}-{x}.csv', index=False)

        loss.append(history)
        Y_test.append(y_test)
        Y_pre.append(pred_y)
        weights.append(list(model.coef_))
        print('第{}年完成'.format(year))

    analysis_data = data[(data['ym'] >= f'{start_year}-01-01') & (data['ym'] <= f'{end_year}-12-31')].loc[:,
                    ['ym', y_col[0]]]
    pre = np.vstack(Y_pre)
    analysis_data['pred'] = pre.reshape(len(pre), )

    R2 = cal_R2(y_test=analysis_data[y_col[0]], pred_y=analysis_data['pred'])
    print('最终ENET的R方：', R2)

    long_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] >= x['pred'].quantile(0.9)])
    short_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] <= x['pred'].quantile(0.1)])
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
else:
    print('ENET不在train_model_list中，跳过训练')

# RF
if 'RF' in train_model_list:
    loss, Y_test, Y_pre = [], [], []
    for year in range(start_year, end_year + 1):
        x_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, x_col + macro_col]
        y_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, y_col[0]]

        x_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:,
                x_col + macro_col]
        y_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        x_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                 x_col + macro_col]
        y_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:, y_col[0]]

        x_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, x_col + macro_col]
        y_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
        test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
        ps = PredefinedSplit(test_fold=test_fold)

        model = random_forest(ps=ps)
        history = model.fit(x_train_val, y_train_val)

        pred_y = model.predict(x_test).reshape(len(y_test), 1)
        print('预测值为:', pred_y)

        R2 = cal_R2(y_test=y_test, pred_y=pred_y)
        print(f'{year}年RF的R方：', R2)

        saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                   ['permno', 'ym', 'sic2', y_col[0]]]
        saved_df['y_pred'] = pred_y

        if not os.path.exists(data_path + f'prediction/{how_to_feature_extraction}'):
            os.mkdir(data_path + f'prediction/{how_to_feature_extraction}')
        saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/RF-{year}.csv', index=False)

        # 因子重要性
        if how_to_feature_extraction == 'origin':
            pass
        else:
            for x in x_col:
                zero_x_test = x_test.copy()
                zero_x_test[x] = 0
                zero_pred_y = model.predict(zero_x_test).reshape(len(y_test), 1)
                saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                           ['permno', 'ym', 'sic2', y_col[0]]]
                saved_df['y_pred'] = zero_pred_y
                saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/RF-{year}-{x}.csv', index=False)

        loss.append(history)
        Y_test.append(y_test)
        Y_pre.append(pred_y)
        print('第{}年完成'.format(year))

    analysis_data = data[(data['ym'] >= f'{start_year}-01-01') & (data['ym'] <= f'{end_year}-12-31')].loc[:, ['ym', y_col[0]]]
    pre = np.vstack(Y_pre)
    analysis_data['pred'] = pre.reshape(len(pre), )

    R2 = cal_R2(y_test=analysis_data[y_col[0]], pred_y=analysis_data['pred'])
    print('最终RF的R方：', R2)

    long_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] >= x['pred'].quantile(0.9)])
    short_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] <= x['pred'].quantile(0.1)])
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
else:
    print('RF不在train_model_list中，跳过训练')

# NN3
if 'NN3' in train_model_list:
    loss, Y_test, Y_pre = [], [], []
    for year in range(start_year, end_year + 1):
        x_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, x_col + macro_col]
        y_train = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 13}-12-31')].loc[:, y_col[0]]

        x_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:,
                x_col + macro_col]
        y_val = data[(data['ym'] >= f'{year - 12}-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        x_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                 x_col + macro_col]
        y_test = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:, y_col[0]]

        x_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, x_col + macro_col]
        y_train_val = data[(data['ym'] >= '1965-01-01') & (data['ym'] <= f'{year - 1}-12-31')].loc[:, y_col[0]]

        # gridsearch
        inpu = Input(int(x_train.shape[1]))
        lr_list = [0.01, 0.001]  #
        l1_list = [0.001, 0.0001]  #
        batch_size_list = [10000]

        pred_y_list = []
        model_list = []
        score_list = []
        for lr in lr_list:
            for l1 in l1_list:
                for batch_size in batch_size_list:
                    mini_model_list = []
                    for ensemble in range(10):
                        model = NN3(inpu=inpu, l1=l1, seed=ensemble)
                        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                                      loss='mse')
                        model.fit(x_train, y_train,
                                  batch_size=batch_size,
                                  epochs=100,
                                  validation_data=(x_val, y_val),
                                  callbacks=my_callbacks,
                                  shuffle=True,
                                  verbose=0
                                  )
                        pred_y = model.predict(x_val).reshape(-1, 1)
                        score_list.append(pred_y)
                        mini_model_list.append(model)
                    pred_y = np.hstack(score_list)
                    pred_y = np.mean(pred_y, axis=1).reshape(-1, 1)

                    score_list.append(cal_R2(y_test=y_val, pred_y=pred_y))
                    model_list.append(mini_model_list)

        best_model_index = score_list.index(max(score_list))

        pred_y_list = []
        for model in model_list[best_model_index]:
            pred_y_list.append(model.predict(x_test).reshape(-1, 1))
        pred_y = np.hstack(pred_y_list)
        pred_y = np.mean(pred_y, axis=1).reshape(-1, 1)
        print('预测值为:', pred_y)

        R2 = cal_R2(y_test=y_test, pred_y=pred_y)
        print(f'{year}年NN3的R方：', R2)

        saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                   ['permno', 'ym', 'sic2', y_col[0]]]
        saved_df['y_pred'] = pred_y

        if not os.path.exists(data_path + f'prediction/{how_to_feature_extraction}'):
            os.mkdir(data_path + f'prediction/{how_to_feature_extraction}')
        saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/NN3-{year}.csv', index=False)

        # 因子重要性
        if how_to_feature_extraction == 'origin':
            pass
        else:
            for x in x_col:
                zero_x_test = x_test.copy()
                zero_x_test[x] = 0
                zero_pred_y = model.predict(zero_x_test).reshape(len(y_test), 1)
                saved_df = data[(data['ym'] >= f'{year}-01-01') & (data['ym'] <= f'{year}-12-31')].loc[:,
                           ['permno', 'ym', 'sic2', y_col[0]]]
                saved_df['y_pred'] = zero_pred_y
                saved_df.to_csv(data_path + f'prediction/{how_to_feature_extraction}/NN3-{year}-{x}.csv', index=False)

        loss.append(history)
        Y_test.append(y_test)
        Y_pre.append(pred_y)
        print('第{}年完成'.format(year))

    analysis_data = data[(data['ym'] >= f'{start_year}-01-01') & (data['ym'] <= f'{end_year}-12-31')].loc[:, ['ym', y_col[0]]]
    pre = np.vstack(Y_pre)
    analysis_data['pred'] = pre.reshape(len(pre), )

    R2 = cal_R2(y_test=analysis_data[y_col[0]], pred_y=analysis_data['pred'])
    print('最终NN3的R方：', R2)

    long_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] >= x['pred'].quantile(0.9)])
    short_portfolio = analysis_data.groupby('ym').apply(lambda x: x[x['pred'] <= x['pred'].quantile(0.1)])
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
else:
    print('NN3不在train_model_list中，跳过训练')

print('all done')