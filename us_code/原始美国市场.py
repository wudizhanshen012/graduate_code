# %%
import tensorflow as tf

def init_cuda():
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if len(gpus) != 0:
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)


init_cuda()

# %%
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from sklearn import preprocessing
import datetime

# %%
data = pd.read_csv("/20230328代码与数据/美国市场数据/final_data——1965-2020.csv", encoding='utf-8')
data['ym'] = data['DATE'].astype(str).apply(lambda x: x[0:4] + '-' + x[4:6])
data = data.set_index(data['ym'])
data.index = pd.to_datetime(data.index)
del data['DATE']
del data['permno']
del data['ym']

data = data.rename(columns={'R-1': 'ret+1'})
data


# %%
def industry_dummy(industry_codes):
    """
    输入一列A股市场各个股票的行业代码，返回由各个股票的行业虚拟变量的向量组成的矩阵
    """
    import pandas as pd
    industry_df = pd.DataFrame(industry_codes, columns=['industry_code'])
    industry_df = pd.get_dummies(industry_df, columns=['industry_code'])
    return industry_df.values


# %%
ind_x = industry_dummy(data['sic2'].values)
ind_x
# %%
# x,y
x_col = list(data.columns)
x_col.remove('RET')
x_col.remove('sic2')
x_col.remove('ret+1')
x = data.loc[:, x_col]
y_col = ['ret+1']
y = data.loc[:, y_col]

x.index, y.index = pd.to_datetime(x.index), pd.to_datetime(y.index)
x, y
# %%
df = pd.read_csv("/20230328代码与数据/美国市场数据/PredictorData2021.xlsx - Monthly.csv")
df['dp'] = np.log(df['D12'] / df['Index'])  # d/p
df['ep'] = np.log(df['E12'] / df['Index'])  # e/p
df['bm'] = df['b/m']  # B/M
df['ntis'] = df['ntis']  # net equity expansion
df['tbl'] = df['tbl']  # Treasury-bill rate
df['tms'] = df['lty'] - df['tbl']  # term-spread
df['dfy'] = df['BAA'] - df['AAA']  # default-spread
df['svar'] = df['svar']  # stock variance

# rf
rf = df.loc[:, ['yyyymm', 'tbl']]
rf['rf+1'] = rf['tbl'].shift(-1)
rf = rf[rf['yyyymm'] >= 196501][df['yyyymm'] <= 202012]
rf = rf.loc[:, ['yyyymm', 'rf+1']]
rf['rf+1'] = rf['rf+1'] / 12
tbl_rf = np.array(rf['rf+1'])

# 差分
df.iloc[:, 1:] = df.iloc[:, 1:].pct_change()# - df.iloc[:, 1:].shift(1)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df=df.fillna(method='ffill')
df = df.fillna(0)
print('-----------------------------------------这个跑的是差分+过去标准化----------------------------------------------------')
df = df.set_index('yyyymm')
mean = df.rolling(window=72, min_periods=72).mean()
std = df.rolling(window=72, min_periods=72).std()
df = (df - mean) / std
df = df.reset_index()

df = df[df['yyyymm'] >= 196501][df['yyyymm'] <= 202012]
df = df.loc[:, ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']]
col = df.columns

# # 宏观数据标准化
# df = 2 * MinMax.fit_transform(df) - 1
# df = pd.DataFrame(df)
# df.columns = col
# print(df)

'''九个宏观数据'''
svar = np.array(df['svar']).reshape(672, 1)
tbl = np.array(df['tbl']).reshape(672, 1)
tms = np.array(df['tms']).reshape(672, 1)
dfy = np.array(df['dfy']).reshape(672, 1)
ep = np.array(df['ep']).reshape(672, 1)
dp = np.array(df['dp']).reshape(672, 1)
ntis = np.array(df['ntis']).reshape(672, 1)
bm = np.array(df['bm']).reshape(672, 1)
# %%
# cross-sectional rank and calcalate excess return
# add macro informations
MinMax = preprocessing.MinMaxScaler()
x_final = []
y_final = []
for year in range(56):
    for month in range(12):
        if year == 55 and month == 11:
            pass
        else:
            x_min = x['{}-{}'.format(1965 + year, month + 1)]
            y_min = y['{}-{}'.format(1965 + year, month + 1)]
            rf = tbl_rf[year * 12 + month]
            x_min.iloc[:, :] = 2 * MinMax.fit_transform(x_min) - 1
            y_min = y_min['ret+1'] - rf
            x_min['DP'] = np.zeros((len(x['{}-{}'.format(1965 + year, month + 1)]),)) + dp[year * 12 + month]
            x_min['EP'] = np.zeros((len(x['{}-{}'.format(1965 + year, month + 1)]),)) + ep[year * 12 + month]
            x_min['BM'] = np.zeros((len(x['{}-{}'.format(1965 + year, month + 1)]),)) + bm[year * 12 + month]
            x_min['NTIS'] = np.zeros((len(x['{}-{}'.format(1965 + year, month + 1)]),)) + ntis[year * 12 + month]
            x_min['TBL'] = np.zeros((len(x['{}-{}'.format(1965 + year, month + 1)]),)) + tbl[year * 12 + month]
            x_min['TMS'] = np.zeros((len(x['{}-{}'.format(1965 + year, month + 1)]),)) + tms[year * 12 + month]
            x_min['DFY'] = np.zeros((len(x['{}-{}'.format(1965 + year, month + 1)]),)) + dfy[year * 12 + month]
            x_min['SVAR'] = np.zeros((len(x['{}-{}'.format(1965 + year, month + 1)]),)) + svar[year * 12 + month]
            x_final.append(x_min)
            y_final.append(y_min)
x, y = pd.concat(x_final), pd.concat(y_final)
# %%
# 美国市场
size = ['mvel1', 'mve_ia', 'herf', 'chinv', 'chcsho', 'age']
beta = ['beta', 'betasq']
mom = ['mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom', 'maxret', 'indmom']
liq = ['std_dolvol', 'zerotrade', 'std_turn', 'ill', 'turn', 'dolvol', 'pricedelay', 'chatoia', 'salerec']
vol = ['retvol', 'aeavol', 'idiovol', 'ear', 'roavol']
bpr = ['bm', 'bm_ia', 'cfp', 'cfp_ia', 'sp', 'invest', 'realestate', 'depr', 'cashpr']
ey = ['roeq', 'ep', 'roaq', 'divo', 'absacc', 'divi', 'chempia', 'nincr', 'chpmia', 'stdacc', 'chtx', 'cash', 'roic',
      'stdcf', 'dy', 'acc', 'pctacc', 'saleinv', 'operprof', 'pchsale_pchrect', 'salecash', 'tb', 'gma', 'pchdepr']
growth = ['grltnoa', 'egr', 'orgcap', 'sgr', 'pchgm_pchsale', 'rsup', 'pchsaleinv', 'rd_sale', 'rd_mve', 'rd',
          'cinvest', 'pchsale_pchxsga', 'pchsale_pchinvt', 'agr', 'grcapx', 'hire']
lever = ['securedind', 'secured', 'lev', 'pchquick', 'pchcapx_ia', 'lgr', 'quick', 'ps', 'tang', 'currat', 'ms',
         'pchcurrat', 'cashdebt', 'convind']



# %%
# 交乘

macro_data = x[['DP', 'EP', 'BM', 'NTIS', 'TBL', 'TMS', 'DFY', 'SVAR']]
ym = x.index
x = x.iloc[:, :-8]
x_array = [x]
for i in range(8):
    x_array.append(x * np.array(macro_data.iloc[:, i]).reshape(-1, 1))

x = pd.DataFrame(np.hstack(x_array))
x.index = ym
x

# 'sin','baspread',
# %%
# 修改
x = pd.concat([x, pd.DataFrame(ind_x, index=x.index)], axis=1)
x

# 原始

# %%
# OLS

# from sklearn.linear_model import LinearRegression, HuberRegressor
# from sklearn.model_selection import PredefinedSplit
# from sklearn.model_selection import GridSearchCV
#
# loss, Y_test, Y_pre = [], [], []
# # for i in range(16):
# #     a = pd.read_csv(f'F:/xiu预测结果/原始202304/OLS-{i + 1}.csv')
# #     Y_pre.append(a)
# weights = []
# for year in range(19):
#     x_train = x['1965':'{}'.format(1988 + year)]
#     y_train = y['1965':'{}'.format(1988 + year)]
#
#     x_val = x['{}'.format(1989 + year):'{}'.format(2000 + year)]
#     y_val = y['{}'.format(1989 + year):'{}'.format(2000 + year)]
#
#     x_test = x['{}'.format(2001 + year)]
#     y_test = y['{}'.format(2001 + year)]
#
#     x_train_val = x['1965':'{}'.format(2000 + year)]
#     y_train_val = y['1965':'{}'.format(2000 + year)]
#
#     test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
#     test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
#     ps = PredefinedSplit(test_fold=test_fold)
#     param_grid = {'l1': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
#                   }
#     model = HuberRegressor(epsilon=1.0)
#     history = model.fit(x_train_val, y_train_val,
#                         )
#
#     pred_y3 = model.predict(x_test).reshape(len(y_test), 1)
#     print('预测值为:', pred_y3)
#
#     R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
#         sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
#     print('OLS的R方：', R方_RF)
#
#     # pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/美国市场预测结果/原始/OLS-{}.csv'.format(year + 1), index=False)
#
#     loss.append(history)
#     Y_test.append(y_test)
#     Y_pre.append(pred_y3)
#     weights.append(list(model.coef_))
#     print('第{}年完成'.format(year + 1))
#
# data = pd.DataFrame()
# pre = np.array(Y_pre[0]).reshape(len(Y_pre[0]), 1)
# for i in range(1, 19):
#     pre = np.vstack([pre, np.array(Y_pre[i]).reshape(len(Y_pre[i]), 1)])
#
# data['pred'] = pre.reshape(len(pre), )
# data['y_true'] = np.array(y['2001':'2019'])
# data.index = y['2001':'2019'].index
#
# y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
# R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
#     sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
# print('最终OLS的R方：', R方_RF)
#
# print(data)
#
# ret_long = []
# ret_short = []
# for year in range(19):
#     for month in range(12):
#         if year == 19 and month == 11:
#             pass
#         else:
#             df = data['{}-{}'.format(2001 + year, 1 + month)]
#             df = df.sort_values('pred', ascending=False)
#             ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), 1]))
#             ret_short.append(np.mean(df.iloc[int(len(df) / 10 * 9):, 1]))
#             # print('第{}年:'.format(year + 1))
#             # print('多:', ret_long[year * 12 + month], '空:', ret_short[year * 12 + month], '多空：',
#             #       ret_long[year * 12 + month] - ret_short[year * 12 + month], '累计收益:', sum(ret_long) - sum(ret_short))
#             # print('累计多:', sum(ret_long), '累计空:', sum(ret_short))
# print('多空平均超额月收益率：', np.mean(np.array(ret_long) - np.array(ret_short)))
# print('夏普比率:', np.mean(np.array(ret_long) - np.array(ret_short)) * 12 / (
#         np.std(np.array(ret_long) - np.array(ret_short)) * np.power(12, 0.5)))

# %%
from sklearn.linear_model import RidgeCV, SGDRegressor
from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

loss, Y_test, Y_pre = [], [], []
weights = []
for year in range(19):
    x_train = x['1965':'{}'.format(1988 + year)]
    y_train = y['1965':'{}'.format(1988 + year)]

    x_val = x['{}'.format(1989 + year):'{}'.format(2000 + year)]
    y_val = y['{}'.format(1989 + year):'{}'.format(2000 + year)]

    x_test = x['{}'.format(2001 + year)]
    y_test = y['{}'.format(2001 + year)]

    x_train_val = x['1965':'{}'.format(2000 + year)]
    y_train_val = y['1965':'{}'.format(2000 + year)]

    test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
    test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
    ps = PredefinedSplit(test_fold=test_fold)

    # model = RidgeCV(alphas=[0.1, 0.01, 0.001, 0.0001],
    #                 cv=ps)

    param_grid = {'alpha': [0.1, 0.01, 0.001, 0.0001], }
    ENet = SGDRegressor(loss='huber',
                        penalty='l2',
                        #l1_ratio=0.5,
                        epsilon=0.999,
                        learning_rate='adaptive',
                        verbose=0)
    model = GridSearchCV(ENet, param_grid, cv=ps, n_jobs=15)

    history = model.fit(x_train_val, y_train_val,
                        )

    print('第{}年完成'.format(year + 1))

    pred_y3 = model.predict(x_test).reshape(len(y_test), 1)
    print('预测值为:', pred_y3)

    R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
        sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
    print('Ridge的R方：', R方_RF)

    # pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/美国市场预测结果/原始/Ridge-{}.csv'.format(year + 1), index=False)

    loss.append(history)
    Y_test.append(y_test)
    Y_pre.append(pred_y3)
    # weights.append(list(model.coef_))

data = pd.DataFrame()
pre = np.array(Y_pre[0]).reshape(len(Y_pre[0]), 1)
for i in range(1, 19):
    pre = np.vstack([pre, np.array(Y_pre[i]).reshape(len(Y_pre[i]), 1)])

data['pred'] = pre.reshape(len(pre), )
data['y_true'] = np.array(y['2001':'2019'])
data.index = y['2001':'2019'].index

y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
    sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
print('最终Ridge的R方：', R方_RF)

print(data)

ret_long = []
ret_short = []
for year in range(19):
    for month in range(12):
        if year == 19 and month == 11:
            pass
        else:
            df = data['{}-{}'.format(2001 + year, 1 + month)]
            df = df.sort_values('pred', ascending=False)
            ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), 1]))
            ret_short.append(np.mean(df.iloc[int(len(df) / 10 * 9):, 1]))
            # print('第{}年:'.format(year + 1))
            # print('多:', ret_long[year * 12 + month], '空:', ret_short[year * 12 + month], '多空：',
            #       ret_long[year * 12 + month] - ret_short[year * 12 + month], '累计收益:', sum(ret_long) - sum(ret_short))
            # print('累计多:', sum(ret_long), '累计空:', sum(ret_short))
print('多空平均超额月收益率：', np.mean(np.array(ret_long) - np.array(ret_short)))
print('夏普比率:', np.mean(np.array(ret_long) - np.array(ret_short)) * 12 / (
        np.std(np.array(ret_long) - np.array(ret_short)) * np.power(12, 0.5)))
# %%
# from sklearn.linear_model import ElasticNetCV, SGDRegressor
# from sklearn.model_selection import PredefinedSplit
# from sklearn.model_selection import GridSearchCV
#
# loss, Y_test, Y_pre = [], [], []
# weights = []
# for year in range(19):
#     x_train = x['1965':'{}'.format(1988 + year)]
#     y_train = y['1965':'{}'.format(1988 + year)]
#
#     x_val = x['{}'.format(1989 + year):'{}'.format(2000 + year)]
#     y_val = y['{}'.format(1989 + year):'{}'.format(2000 + year)]
#
#     x_test = x['{}'.format(2001 + year)]
#     y_test = y['{}'.format(2001 + year)]
#
#     x_train_val = x['1965':'{}'.format(2000 + year)]
#     y_train_val = y['1965':'{}'.format(2000 + year)]
#
#     test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
#     test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
#     ps = PredefinedSplit(test_fold=test_fold)
#     param_grid = {'l1': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
#                   }
#     model = ElasticNetCV(alphas=[0.001, 0.0001],
#                          l1_ratio=0.5,
#                          cv=ps)
#
#     param_grid = {'alpha': [0.1, 0.01, 0.001, 0.0001], }
#     ENet = SGDRegressor(loss='huber',
#                         penalty='elasticnet',
#                         l1_ratio=0.5,
#                         epsilon=0.999,
#                         learning_rate='adaptive',
#                         verbose=0)
#     model = GridSearchCV(ENet, param_grid, cv=ps, n_jobs=15)
#
#     history = model.fit(x_train_val, y_train_val,
#                         )
#
#     print('第{}年完成'.format(year + 1))
#
#     pred_y3 = model.predict(x_test).reshape(len(y_test), 1)
#     print('预测值为:', pred_y3)
#
#     R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
#         sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
#     print('ENet的R方：', R方_RF)
#
#     # pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/美国市场预测结果/原始/ENet-{}.csv'.format(year + 1), index=False)
#
#     loss.append(history)
#     Y_test.append(y_test)
#     Y_pre.append(pred_y3)
#     # weights.append(list(model.coef_))
#
# data = pd.DataFrame()
# pre = np.array(Y_pre[0]).reshape(len(Y_pre[0]), 1)
# for i in range(1, 19):
#     pre = np.vstack([pre, np.array(Y_pre[i]).reshape(len(Y_pre[i]), 1)])
#
# data['pred'] = pre.reshape(len(pre), )
# data['y_true'] = np.array(y['2001':'2019'])
# data.index = y['2001':'2019'].index
#
# y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
# R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
#     sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
# print('最终ENet的R方：', R方_RF)
#
# print(data)
#
# ret_long = []
# ret_short = []
# for year in range(19):
#     for month in range(12):
#         if year == 19 and month == 11:
#             pass
#         else:
#             df = data['{}-{}'.format(2001 + year, 1 + month)]
#             df = df.sort_values('pred', ascending=False)
#             ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), 1]))
#             ret_short.append(np.mean(df.iloc[int(len(df) / 10 * 9):, 1]))
#             # print('第{}年:'.format(year + 1))
#             # print('多:', ret_long[year * 12 + month], '空:', ret_short[year * 12 + month], '多空：',
#             #       ret_long[year * 12 + month] - ret_short[year * 12 + month], '累计收益:', sum(ret_long) - sum(ret_short))
#             # print('累计多:', sum(ret_long), '累计空:', sum(ret_short))
# print('多空平均超额月收益率：', np.mean(np.array(ret_long) - np.array(ret_short)))
# print('夏普比率:', np.mean(np.array(ret_long) - np.array(ret_short)) * 12 / (
#         np.std(np.array(ret_long) - np.array(ret_short)) * np.power(12, 0.5)))
#
# # %%
# # Lasso
# from sklearn.linear_model import LassoCV
# from sklearn.model_selection import PredefinedSplit
# from sklearn.model_selection import GridSearchCV
#
# loss, Y_test, Y_pre = [], [], []
# weights = []
# for year in range(19):
#     x_train = x['1965':'{}'.format(1988 + year)]
#     y_train = y['1965':'{}'.format(1988 + year)]
#
#     x_val = x['{}'.format(1989 + year):'{}'.format(2000 + year)]
#     y_val = y['{}'.format(1989 + year):'{}'.format(2000 + year)]
#
#     x_test = x['{}'.format(2001 + year)]
#     y_test = y['{}'.format(2001 + year)]
#
#     x_train_val = x['1965':'{}'.format(2000 + year)]
#     y_train_val = y['1965':'{}'.format(2000 + year)]
#
#     test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
#     test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
#     ps = PredefinedSplit(test_fold=test_fold)
#     param_grid = {'l1': [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001],
#                   }
#     # model = LassoCV(alphas=[0.1, 0.01, 0.001, 0.0001],
#     #                 cv=ps)
#     param_grid = {'alpha': [0.001, 0.0001], }
#     ENet = SGDRegressor(loss='huber',
#                         penalty='l1',
#                         #l1_ratio=0.5,
#                         epsilon=0.999,
#                         learning_rate='adaptive',
#                         verbose=0)
#     model = GridSearchCV(ENet, param_grid, cv=ps, n_jobs=15)
#     history = model.fit(x_train_val, y_train_val,
#                         )
#
#     pred_y3 = model.predict(x_test).reshape(len(y_test), 1)
#     print('预测值为:', pred_y3)
#
#     R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
#         sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
#     print('Lasso的R方：', R方_RF)
#
#     # pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/美国市场预测结果/原始/Lasso-{}.csv'.format(year + 1), index=False)
#
#     loss.append(history)
#     Y_test.append(y_test)
#     Y_pre.append(pred_y3)
#     # weights.append(list(model.coef_))
#     print('第{}年完成'.format(year + 1))
#
# data = pd.DataFrame()
# pre = np.array(Y_pre[0]).reshape(len(Y_pre[0]), 1)
# for i in range(1, 19):
#     pre = np.vstack([pre, np.array(Y_pre[i]).reshape(len(Y_pre[i]), 1)])
#
# data['pred'] = pre.reshape(len(pre), )
# data['y_true'] = np.array(y['2001':'2019'])
# data.index = y['2001':'2019'].index
#
# y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
# R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
#     sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
# print('最终Lasso的R方：', R方_RF)
#
# print(data)
#
# ret_long = []
# ret_short = []
# for year in range(19):
#     for month in range(12):
#         if year == 19 and month == 11:
#             pass
#         else:
#             df = data['{}-{}'.format(2001 + year, 1 + month)]
#             df = df.sort_values('pred', ascending=False)
#             ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), 1]))
#             ret_short.append(np.mean(df.iloc[int(len(df) / 10 * 9):, 1]))
#             # print('第{}年:'.format(year + 1))
#             # print('多:', ret_long[year * 12 + month], '空:', ret_short[year * 12 + month], '多空：',
#             #       ret_long[year * 12 + month] - ret_short[year * 12 + month], '累计收益:', sum(ret_long) - sum(ret_short))
#             # print('累计多:', sum(ret_long), '累计空:', sum(ret_short))
# print('多空平均超额月收益率：', np.mean(np.array(ret_long) - np.array(ret_short)))
# print('夏普比率:', np.mean(np.array(ret_long) - np.array(ret_short)) * 12 / (
#         np.std(np.array(ret_long) - np.array(ret_short)) * np.power(12, 0.5)))
# %%
# RF

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestRegressor

loss, Y_test, Y_pre = [], [], []

for year in range(19):
    x_train = x['1965':'{}'.format(1988 + year)]
    y_train = y['1965':'{}'.format(1988 + year)]

    x_val = x['{}'.format(1989 + year):'{}'.format(2000 + year)]
    y_val = y['{}'.format(1989 + year):'{}'.format(2000 + year)]

    x_test = x['{}'.format(2001 + year)]
    y_test = y['{}'.format(2001 + year)]

    x_train_val = x['1965':'{}'.format(2000 + year)]
    y_train_val = y['1965':'{}'.format(2000 + year)]

    test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
    test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
    ps = PredefinedSplit(test_fold=test_fold)
    param_grid = {'max_features': [3, 5, 10],
                  'max_depth': [1, 2],
                  'n_estimators': [50, 100, 200, 300],
                  }
    model = RandomForestRegressor(n_jobs=15)
    model = GridSearchCV(model, param_grid, cv=ps, n_jobs=15
                         )

    history = model.fit(x_train_val, y_train_val)

    print('第{}年完成'.format(year + 1))

    pred_y3 = model.predict(x_test).reshape(len(y_test), 1)
    print('预测值为:', pred_y3)

    # pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/美国市场预测结果/原始/RF-{}.csv'.format(year + 1), index=False)

    R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
        sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
    print('RF的R方：', R方_RF)

    loss.append(history)
    Y_test.append(y_test)
    Y_pre.append(pred_y3)

data = pd.DataFrame()
pre = np.array(Y_pre[0]).reshape(len(Y_pre[0]), 1)
for i in range(1, 19):
    pre = np.vstack([pre, np.array(Y_pre[i]).reshape(len(Y_pre[i]), 1)])

data['pred'] = pre.reshape(len(pre), )
data['y_true'] = np.array(y['2001':'2019'])
data.index = y['2001':'2019'].index

y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
    sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
print('最终RF的R方：', R方_RF)

print(data)

ret_long = []
ret_short = []
for year in range(19):
    for month in range(12):
        if year == 19 and month == 11:
            pass
        else:
            df = data['{}-{}'.format(2001 + year, 1 + month)]
            df = df.sort_values('pred', ascending=False)
            ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), 1]))
            ret_short.append(np.mean(df.iloc[int(len(df) / 10 * 9):, 1]))
            # print('第{}年:'.format(year + 1))
            # print('多:', ret_long[year * 12 + month], '空:', ret_short[year * 12 + month], '多空：',
            #       ret_long[year * 12 + month] - ret_short[year * 12 + month], '累计收益:', sum(ret_long) - sum(ret_short))
            # print('累计多:', sum(ret_long), '累计空:', sum(ret_short))
print('多空平均超额月收益率：', np.mean(np.array(ret_long) - np.array(ret_short)))
print('夏普比率:', np.mean(np.array(ret_long) - np.array(ret_short)) * 12 / (
        np.std(np.array(ret_long) - np.array(ret_short)) * np.power(12, 0.5)))
# %%
# NN3


import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import BatchNormalization, Input

my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
from tensorflow.keras.layers import Average
from tensorflow.keras.models import Model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from tensorflow.keras.layers import Dropout


def CNN(inpu, l1, seed):
    x = Dense(32, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
              kernel_regularizer=tf.keras.regularizers.l1(l1)
              )(inpu)
    x = Dense(16, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
              kernel_regularizer=tf.keras.regularizers.l1(l1))(x)
    x = Dense(8, activation='relu', kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
              kernel_regularizer=tf.keras.regularizers.l1(l1))(x)
    x = Dense(1, activation='linear', kernel_initializer=tf.keras.initializers.he_normal(seed=seed),
              kernel_regularizer=tf.keras.regularizers.l1(l1))(x)
    model = Model(inpu, x)
    return model


def ensemble(models, train_x):
    outputs = [model.outputs[0] for model in models]
    y = Average()(outputs)
    model = Model(train_x, y, name='ensemble')
    return model


inpu = Input(int(x.shape[1]))


def build_model(lr, l1):
    model1, model2, model3, model4, model5, model6, model7, model8, model9, model10 = \
        CNN(inpu, l1, 1), CNN(inpu, l1, 2), CNN(inpu, l1, 3), CNN(inpu, l1, 4), CNN(inpu, l1, 5), \
        CNN(inpu, l1, 6), CNN(inpu, l1, 7), CNN(inpu, l1, 8), CNN(inpu, l1, 9), CNN(inpu, l1, 10)
    models = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]
    NN3 = ensemble(models, inpu)
    NN3.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                loss='mse')
    return NN3


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestRegressor

loss, Y_test, Y_pre = [], [], []
# for i in range(13):
#     Y_pre.append(pd.read_csv('F:/xiu预测结果/原始202304/NN3-{}.csv'.format(i+1)))

for year in range(19):
    x_train = x['1965':'{}'.format(1988 + year)]  # .iloc[:,:17]
    y_train = y['1965':'{}'.format(1988 + year)]

    x_val = x['{}'.format(1989 + year):'{}'.format(2000 + year)]  # .iloc[:,:17]
    y_val = y['{}'.format(1989 + year):'{}'.format(2000 + year)]

    x_test = x['{}'.format(2001 + year)]  # .iloc[:,:17]
    y_test = y['{}'.format(2001 + year)]

    x_train_val = x['1965':'{}'.format(2000 + year)]  # .iloc[:,:17]
    y_train_val = y['1965':'{}'.format(2000 + year)]

    test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
    test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
    ps = PredefinedSplit(test_fold=test_fold)
    # gridsearch
    lr_list = [0.01, 0.001]  #
    l1_list = [0.001, 0.0001]  #
    batch_size_list = [10000]

    pred_y3_list = []
    model_list = []
    score_list = []
    for lr in lr_list:
        for l1 in l1_list:
            for batch_size in batch_size_list:
                mini_model_list = []
                for ensemble in range(10):
                    model = CNN(inpu=inpu, l1=l1, seed=ensemble)
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
                    pred_y3 = model.predict(x_val).reshape(-1, 1)
                    score_list.append(pred_y3)
                    mini_model_list.append(model)
                pred_y3 = np.hstack(score_list)
                pred_y3 = np.mean(pred_y3, axis=1).reshape(-1, 1)

                score_list.append(1 - (sum((np.array(y_val).reshape(len(y_val), 1) - pred_y3) ** 2)) / (
        sum((np.array(y_val).reshape(len(y_val), 1) ** 2))))
                model_list.append(mini_model_list)

    best_model_index = score_list.index(max(score_list))

    pred_y3_list = []
    for model in model_list[best_model_index]:
        pred_y3_list.append(model.predict(x_test).reshape(-1, 1))
    pred_y3 = np.hstack(pred_y3_list)
    pred_y3 = np.mean(pred_y3, axis=1).reshape(-1, 1)

    R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
        sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
    print('NN3的R方：', R方_RF)
    print('第{}年完成'.format(year + 1))

    loss.append(model)
    Y_test.append(y_test)
    Y_pre.append(pred_y3)

    # pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/美国市场预测结果/原始/NN3-{}.csv'.format(year + 1), index=False)

data = pd.DataFrame()
pre = np.array(Y_pre[0]).reshape(len(Y_pre[0]), 1)
for i in range(1, 19):
    pre = np.vstack([pre, np.array(Y_pre[i]).reshape(len(Y_pre[i]), 1)])

data['pred'] = pre.reshape(len(pre), )
data['y_true'] = np.array(y['2001':'2019'])
data.index = y['2001':'2019'].index

y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
    sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
print('最终NN3的R方：', R方_RF)

print(data)

ret_long = []
ret_short = []
for year in range(19):
    for month in range(12):
        if year == 19 and month == 11:
            pass
        else:
            df = data['{}-{}'.format(2001 + year, 1 + month)]
            df = df.sort_values('pred', ascending=False)
            ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), 1]))
            ret_short.append(np.mean(df.iloc[int(len(df) / 10 * 9):, 1]))
            # print('第{}年:'.format(year + 1))
            # print('多:', ret_long[year * 12 + month], '空:', ret_short[year * 12 + month], '多空：',
            #       ret_long[year * 12 + month] - ret_short[year * 12 + month], '累计收益:', sum(ret_long) - sum(ret_short))
            # print('累计多:', sum(ret_long), '累计空:', sum(ret_short))
print('多空平均超额月收益率：', np.mean(np.array(ret_long) - np.array(ret_short)))
print('夏普比率:', np.mean(np.array(ret_long) - np.array(ret_short)) * 12 / (
        np.std(np.array(ret_long) - np.array(ret_short)) * np.power(12, 0.5)))

