# %%
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')
from sklearn import preprocessing
import datetime

# %%
data = pd.read_csv("/20230328代码与数据/中国市场数据/final_data_china_2022-12-22.csv")
data = data.iloc[:, 1:]
data.replace(np.inf, 0, inplace=True)
data.replace(-np.inf, 0, inplace=True)
data
# %%
# 剔除st和金融股
data = data[data['Ind'] != 'J']

st_finance = pd.read_excel("/20230328代码与数据/中国市场数据/st与行业代码.xlsx", engine='openpyxl')
st_finance['Enddate'] = pd.to_datetime(st_finance['Enddate'])
st_finance['y'] = st_finance['Enddate'].apply(lambda x: x.year)
st_finance = st_finance.loc[:, ['Symbol', 'STPT', 'y']].rename(columns={'Symbol': 'Stkcd'})
st_finance['Stkcd'] = st_finance['Stkcd'].astype('int64')

# 合并
data = pd.merge(data, st_finance, on=['Stkcd', 'y'], how='left')

# 剔除STPT
data = data[data['STPT'] != 1]
del data['STPT']
data
# %%
# 季度频率数据：
quarter_data = ['bm', 'bm_ia', 'cash', 'cashdebt', 'cfp', 'cfp_ia', 'chato', 'chato_ia', 'chinv', 'chpm', 'chpm_ia',
                'chtx', 'cinvest', 'currat', 'ear', 'egr', 'gma', 'herf', 'lev', 'lgr', 'nincr', 'operprof', 'orgcap',
                'pchcapx_ia', 'pchcurrat', 'pchgm_pchsale', 'pchquick', 'pchsale_pchinvt', 'pchsale_pchrect',
                'pchsale_pchxsga', 'pchsaleinv', 'ps', 'quick', 'rd', 'rd_mve', 'rd_sale', 'roaq', 'roavol', 'roeq',
                'roic', 'rsup', 'salecash', 'saleinv', 'salerev', 'sgr', 'sp', 'stdacc', 'stdcf', 'tang', 'tb']
# 半年度频率数据：
half_year_data = ['acc', 'absacc', 'depr', 'pchdepr', 'pchsaleinv']
# 年度频率数据：
year_data = ['agr', 'chempia', 'divi', 'divo', 'dy', 'grCAPX', 'hire', 'invest', 'ms']
data[quarter_data] = data.groupby('Stkcd')[quarter_data].shift(3)
data[half_year_data] = data.groupby('Stkcd')[half_year_data].shift(3)
data[year_data] = data.groupby('Stkcd')[year_data].shift(6)
data
# %%
for i in range(90):
    print(data.iloc[:, 7 + i].describe())
    data.iloc[:, 7 + i] = np.where(data.iloc[:, 7 + i] <= data.iloc[:, 7 + i].quantile(0.05),
                                   data.iloc[:, 7 + i].quantile(0.05),
                                   np.where(data.iloc[:, 7 + i] >= data.iloc[:, 7 + i].quantile(0.95),
                                            data.iloc[:, 7 + i].quantile(0.95), data.iloc[:, 7 + i]))

# %%
macro_data = pd.read_csv("/20230328代码与数据/中国市场数据/macro_data.csv")
macro_data
# %%
macro_data = macro_data.loc[:,
             ['y', 'm', 'dp', 'de', 'bm', 'ep', 'ntis', 'svar', 'tms', 'mtr', 'CPI', 'M2_growth', 'itgr']].rename(
    columns={'CPI': 'infl',
             'M2_growth': 'm2gr'}
    )
macro_data.iloc[:, 2:] = macro_data.iloc[:, 2:].astype('float')
# 均值填充
# for i in macro_data.iloc[:,2:].columns:
#    macro_data[i] = macro_data[i].fillna(macro_data[i].mean())
macro_data.fillna(0, inplace=True)

# 差分
macro_data.iloc[:, [2, 5, 6, 9, 11, 12]] = macro_data.iloc[:, [2, 5, 6, 9, 11, 12]].fillna(0).diff() / macro_data.iloc[
                                                                                                       :,
                                                                                                       [2, 5, 6, 9, 11,
                                                                                                        12]].shift(1)
# 二次差分
macro_data.iloc[:, [5]] = macro_data.iloc[:, [5]].fillna(0).diff() / macro_data.iloc[:, [5]].shift(1)
# 本期减上期
macro_data.iloc[:, [7]] = macro_data.iloc[:, [7]].fillna(0).diff()
macro_data.iloc[:, [7]] = macro_data.iloc[:, [7]].fillna(0).diff()
# CPI
macro_data.iloc[:, [10]] = macro_data.iloc[:, [10]].fillna(0).diff()
macro_data.iloc[:, [10]] = macro_data.iloc[:, [10]].fillna(0).diff()

# bm
macro_data.iloc[:, [4]] = macro_data.iloc[:, [4]].fillna(0).diff() / macro_data.iloc[:, [4]].shift(1)
macro_data.iloc[:, [4]] = macro_data.iloc[:, [4]].fillna(0).diff() / macro_data.iloc[:, [4]].shift(1)

# tms
macro_data.iloc[:, [8]] = macro_data.iloc[:, [8]].fillna(0).diff() / macro_data.iloc[:, [8]].shift(1)
macro_data.iloc[:, [8]] = macro_data.iloc[:, [8]].fillna(0).diff() / macro_data.iloc[:, [8]].shift(1)

macro_data.replace(np.inf, 0, inplace=True)
macro_data.replace(-np.inf, 0, inplace=True)

# 单位化
MinMax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
macro_data.iloc[:, 2:] = MinMax.fit_transform(macro_data.iloc[:, 2:])

macro_data.fillna(0, inplace=True)

macro_data
# %%
final_data = pd.merge(data, macro_data, on=['y', 'm'], how='left')
final_data = final_data[(final_data['y'] >= 2000)]
final_data
# %%
# data standardization
MinMax = preprocessing.MinMaxScaler(feature_range=(-1, 1))
final_data.replace(np.inf, 0, inplace=True)
final_data.replace(-np.inf, 0, inplace=True)
final_data.iloc[:, 7:-11] = final_data.fillna(0).groupby(['y', 'm']).apply(
    lambda x: pd.DataFrame(MinMax.fit_transform(x.iloc[:, 7:-11]), columns=x.iloc[:, 7:-11].columns)).values
final_data = final_data.fillna(0)
final_data
# %%
# 计算超额收益率
risk_free_ret = pd.read_csv('/20230328代码与数据/中国市场数据/risk_free_ret.csv')
risk_free_ret.iloc[:, -1] = risk_free_ret.iloc[:, -1] / 100

final_data = pd.merge(final_data, risk_free_ret, on=['y', 'm'], how='left')
final_data['risk_free_ret_t+1'] = final_data['risk_free_ret_t+1'].fillna(final_data['risk_free_ret_t+1'].mean())
final_data['ret+1'] = final_data['ret+1'] - final_data['risk_free_ret_t+1'] / 12
del final_data['risk_free_ret_t+1']
final_data


# %%
# 行业虚拟变量
def industry_dummy(industry_codes):
    """
    输入一列A股市场各个股票的行业代码，返回由各个股票的行业虚拟变量的向量组成的矩阵
    """
    import pandas as pd
    industry_df = pd.DataFrame(industry_codes, columns=['industry_code'])
    industry_df = pd.get_dummies(industry_df, columns=['industry_code'])
    return industry_df.values


ind = pd.read_csv("/20230328代码与数据/中国市场数据/行业代码_treated.csv")
ind['Stkcd'] = ind['Stkcd'].astype('int64')
ind_x = industry_dummy(pd.merge(final_data, ind, on=['Stkcd'], how='left')['sic2'].values)
ind_x.shape, ind_x
# %%
# 其他虚拟变量
dummy1 = pd.read_csv("/20230328代码与数据/中国市场数据/EN_treated.csv").rename(columns={'year': 'y'})
final_data = pd.merge(final_data, dummy1, on=['Stkcd', 'y'], how='left')

final_data
# %%
x_columns = ['y', 'm'] + list(final_data.iloc[:, 7:].columns)

x = final_data.loc[:, x_columns].rename(columns={'bm_y': 'bm_macro',
                                                 'bm_x': 'bm'}).fillna(0)
y = final_data.iloc[:, [2, 3, 5]]
x, y
# %%
x

# AE
train = False
if train:
    # %%
    size = ['mve', 'mve_ia', 'herf', 'chinv', 'chcsho']
    beta = ['beta', 'betasq']
    mom = ['mom1m', 'mom6m', 'mom12m', 'mom36m', 'chmom', 'maxret', 'er_trend']
    liq = ['std_dolvol', 'zerotrade', 'atr', 'std_turn', 'ill', 'turn', 'dolvol', 'pricedelay', 'chato_ia']
    vol = ['volatility', 'idiovol', 'ear', 'roavol']
    bpr = ['bm', 'bm_ia', 'cfp', 'cfp_ia', 'sp', 'invest', 'realestate', 'depr', 'cashspr']
    ey = 'roeq, roaq, divo, absacc, divi, salerev, chempia, nincr, chpm_ia, stdacc, chtx, cash, roic, chpm, stdcf, chato, dy, acc, pctacc, saleinv, operprof, pchsale_pchrect, salecash, tb, gma, pchdepr'.split(
        ', ')
    growth = 'egr, orgcap, sgr, pchgm_pchsale, rsup, pchsaleinv, rd_sale, rd_mve, rd, cinvest, pchsale_pchxsga, pchsale_pchinvt, agr, grCAPX, hire'.split(
        ', ')
    lever = 'lev, pchquick, pchcapx_ia, lgr, quick, ps, tang, currat, ms, pchcurrat, cashdebt'.split(', ')
    # %%
    name = ['size', 'beta', 'mom', 'liq', 'vol', 'bpr', 'ey', 'growth', 'lever']

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

    AE_data = pd.DataFrame()
    fuzhu = 0
    for i in [size, beta, mom, liq, vol, bpr, ey, growth, lever]:  #
        x1 = x.loc[:, i]

        x_train = x1[x['y'] <= 2008]

        x_val = x1[(x['y'] >= 2009) & (x['y'] <= 2011)]

        x_test = x1[(x['y'] >= 2012) & (x['y'] <= 2022)]

        x_train_val = x1[x['y'] <= 2011]

        M = []
        M_final = []
        lr_list = [0.01, 0.001]
        neuron_list = [4, 16, 64]
        dropout_list = [0, 0.2, 0.5]
        neuron2_list = [4, 16, 64]
        for lr in lr_list:
            for neuron in neuron_list:
                for dropout in dropout_list:
                  for neuron2 in neuron2_list:
                    # in_cell = 5
                    input_dim = np.array(x_train).shape[1]
                    latent_dim = 1
                    batch_size = 10000
                    my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]

                    inputs = Input(shape=(input_dim))

                    encoded = Dense(neuron, activation='tanh')(inputs)
                    # encoded = BatchNormalization()(encoded)
                    encoded = Dropout(dropout)(encoded)
                    encoded = Dense(neuron2, activation='tanh')(encoded)
                    encoded = Dropout(dropout)(encoded)
                    encoded = Dense(latent_dim, activation='tanh')(encoded)

                    # decoded = BatchNormalization()(encoded)
                    decoded = Dense(neuron2, activation='tanh')(encoded)
                    # decoded = BatchNormalization()(decoded)
                    decoded = Dropout(dropout)(decoded)
                    decoded = Dense(neuron, activation='tanh')(decoded)
                    # decoded = BatchNormalization()(decoded)
                    decoded = Dropout(dropout)(decoded)
                    decoded = Dense(input_dim, activation='tanh')(decoded)

                    encoder = Model(inputs, encoded)
                    autoencoder = Model(inputs, decoded)
                    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')  #

                    autoencoder.fit(x_train, x_train, epochs=100, batch_size=batch_size,
                                    validation_data=(x_val, x_val),
                                    callbacks=my_callbacks)

                    M.append(autoencoder)
                    M_final.append(encoder)
        val_mse = []
        for i in M:
            pred = i.predict(x_val)
            val = np.array(x_val)
            mse = ((pred - val) ** 2).mean()
            val_mse.append(mse)
        position = val_mse.index(min(val_mse))
        encoder = M_final[position]

        AE_data[name[fuzhu]] = encoder.predict(x1).reshape(len(x1), )
        print(name[fuzhu], 'done!')
        fuzhu += 1
    # %%
    col = name + ['largestholderrate', 'top10holderrate', 'soe', 'private', 'foreign', 'others']

    AE_data = np.hstack(
        [AE_data, x[['largestholderrate', 'top10holderrate', 'soe', 'private', 'foreign', 'others']].values])

    AE_data = pd.DataFrame(AE_data)
    AE_data.index = x.index
    AE_data.columns = col
    AE_data['ret+1'] = y['ret+1'].values
    AE_data['y'] = x['y'].values
    AE_data['m'] = x['m'].values
    AE_data = AE_data[['y', 'm', 'ret+1'] + col]
    AE_data.to_csv('/20230328代码与数据/中国市场数据/AE_data.csv', index=False)

    AE_data = AE_data[col]
else:
    AE_data = pd.read_csv('/20230328代码与数据/中国市场数据/AE_data.csv')
    name = ['size', 'beta', 'mom', 'liq', 'vol', 'bpr', 'ey', 'growth', 'lever']
    col = name + ['largestholderrate', 'top10holderrate', 'soe', 'private', 'foreign', 'others']
    AE_data = AE_data[col]

# %%
# 交乘

macro_data = x.iloc[:, -15:-4]
ym = x.iloc[:, :2]
x_array = [AE_data]
for i in range(11):
    x_array.append(AE_data * np.array(macro_data.iloc[:, i]).reshape(-1, 1))

x = pd.DataFrame(np.hstack([ym] + x_array))
x.columns = ['y', 'm'] + list(np.arange(x.shape[1] - 2))
x
# %%
# 添加行业虚拟变量
x = pd.DataFrame(np.hstack([x, ind_x]), columns=['y', 'm'] + list(np.arange(x.shape[1] + ind_x.shape[1] - 2)))
x

# %%
from sklearn.linear_model import LinearRegression, HuberRegressor

from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

loss, Y_test, Y_pre = [], [], []
weights = []
for year in range(10):
    x_train = x[x['y'] <= 2008 + year].iloc[:, 2:]
    y_train = y[y['y'] <= 2008 + year].iloc[:, -1]

    x_val = x[(x['y'] >= 2009 + year) & (x['y'] <= 2011 + year)].iloc[:, 2:]
    y_val = y[(y['y'] >= 2009 + year) & (y['y'] <= 2011 + year)].iloc[:, -1]

    x_test = x[x['y'] == 2012 + year].iloc[:, 2:]
    y_test = y[y['y'] == 2012 + year].iloc[:, -1]

    x_train_val = x[x['y'] <= 2011 + year].iloc[:, 2:]
    y_train_val = y[y['y'] <= 2011 + year].iloc[:, -1]

    test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
    test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
    ps = PredefinedSplit(test_fold=test_fold)
    model = HuberRegressor(epsilon=1.35)
    history = model.fit(x_train_val, y_train_val,
                        )

    print('第{}年完成'.format(year + 1))

    pred_y3 = model.predict(x_test).reshape(len(y_test), 1)
    print('预测值为:', pred_y3)

    R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
        sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
    print('OLS的R方：', R方_RF)

    pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/中国市场预测结果/AE/OLS-{}.csv'.format(year + 1), index=False)

    # 因子重要性
    col = ['size', 'beta', 'mom', 'liq', 'vol', 'bpr', 'ey', 'growth', 'lever']
    for i in range(9):
        x2 = x_test.copy()
        feature = col[i]
        for j in range(11):
            x2.iloc[:, i + (j * 15)] = 0
        pd.DataFrame(
            model.predict(x2)).to_csv('/20230328代码与数据/中国市场预测结果/AE/OLS-{}-{}.csv'.format(year + 1, feature),
                                      index=False)

    loss.append(history)
    Y_test.append(y_test)
    Y_pre.append(pred_y3)
    weights.append(list(model.coef_))

data = pd.DataFrame()
pre = Y_pre[0]
for i in range(1, 10):
    pre = np.vstack([pre, Y_pre[i]])

data['pred'] = pre.reshape(len(pre), )
data['y'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'y'].values
data['m'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'm'].values
data['y_true'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'ret+1'].values

y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
R2_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
    sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
print('最终OLS的R方：', R2_RF)

print(data)

ret_long = []
ret_short = []
for year in range(10):
    for month in range(12):
        df = data[(data['y'] == 2012 + year) & (data['m'] == 1 + month)]
        df = df.sort_values('pred', ascending=False)
        ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), -1]))
        ret_short.append(np.mean(df.iloc[-int(len(df) / 10):, -1]))
        # print('第{}年:'.format(year + 1))
        # print('多:', ret_long[year * 12 + month], '空:', ret_short[year * 12 + month], '多空：',
        #       ret_long[year * 12 + month] - ret_short[year * 12 + month], '累计收益:', sum(ret_long) - sum(ret_short))
        # print('累计多:', sum(ret_long), '累计空:', sum(ret_short))
print('多空平均超额月收益率：', np.mean(np.array(ret_long) - np.array(ret_short)))
print('夏普比率:', np.mean(np.array(ret_long) - np.array(ret_short)) * 12 / (
            np.std(np.array(ret_long) - np.array(ret_short)) * np.power(12, 0.5)))
# %%
from sklearn.linear_model import LassoCV, SGDRegressor

from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

loss, Y_test, Y_pre = [], [], []
weights = []
for year in range(10):
    x_train = x[x['y'] <= 2008 + year].iloc[:, 2:]
    y_train = y[y['y'] <= 2008 + year].iloc[:, -1]

    x_val = x[(x['y'] >= 2009 + year) & (x['y'] <= 2011 + year)].iloc[:, 2:]
    y_val = y[(y['y'] >= 2009 + year) & (y['y'] <= 2011 + year)].iloc[:, -1]

    x_test = x[x['y'] == 2012 + year].iloc[:, 2:]
    y_test = y[y['y'] == 2012 + year].iloc[:, -1]

    x_train_val = x[x['y'] <= 2011 + year].iloc[:, 2:]
    y_train_val = y[y['y'] <= 2011 + year].iloc[:, -1]

    test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
    test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
    ps = PredefinedSplit(test_fold=test_fold)
    # model = LassoCV(alphas=[0.001, 0.0001],
    #                 cv=ps)

    param_grid = {'alpha':[0.001,0.0001]}
    ENet = SGDRegressor(loss='huber',
                        penalty='l1',
                        # l1_ratio=0.5,
                        epsilon=1.35,
                        learning_rate='adaptive',
                        verbose=0)
    model = GridSearchCV(ENet, param_grid, cv = ps)

    history = model.fit(x_train_val, y_train_val,
                        )

    print('第{}年完成'.format(year + 1))

    pred_y3 = model.predict(x_test).reshape(len(y_test), 1)
    print('预测值为:', pred_y3)

    R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
        sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
    print('Lasso的R方：', R方_RF)

    pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/中国市场预测结果/AE/Lasso-{}.csv'.format(year + 1), index=False)

    # 因子重要性
    col = ['size', 'beta', 'mom', 'liq', 'vol', 'bpr', 'ey', 'growth', 'lever']
    for i in range(9):
        x2 = x_test.copy()
        feature = col[i]
        for j in range(11):
            x2.iloc[:, i + (j * 15)] = 0
        pd.DataFrame(
            model.predict(x2)).to_csv('/20230328代码与数据/中国市场预测结果/AE/Lasso-{}-{}.csv'.format(year + 1, feature),
                                      index=False)

    loss.append(history)
    Y_test.append(y_test)
    Y_pre.append(pred_y3)
    # weights.append(list(model.coef_))

data = pd.DataFrame()
pre = Y_pre[0]
for i in range(1, 10):
    pre = np.vstack([pre, Y_pre[i]])

data['pred'] = pre.reshape(len(pre), )
data['y'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'y'].values
data['m'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'm'].values
data['y_true'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'ret+1'].values

y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
R2_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
    sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
print('最终Lasso的R方：', R2_RF)


ret_long = []
ret_short = []
for year in range(10):
    for month in range(12):
        df = data[(data['y'] == 2012 + year) & (data['m'] == 1 + month)]
        df = df.sort_values('pred', ascending=False)
        ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), -1]))
        ret_short.append(np.mean(df.iloc[-int(len(df) / 10):, -1]))
        # print('第{}年:'.format(year + 1))
        # print('多:', ret_long[year * 12 + month], '空:', ret_short[year * 12 + month], '多空：',
        #       ret_long[year * 12 + month] - ret_short[year * 12 + month], '累计收益:', sum(ret_long) - sum(ret_short))
        # print('累计多:', sum(ret_long), '累计空:', sum(ret_short))
print('多空平均超额月收益率：', np.mean(np.array(ret_long) - np.array(ret_short)))
print('夏普比率:', np.mean(np.array(ret_long) - np.array(ret_short)) * 12 / (
            np.std(np.array(ret_long) - np.array(ret_short)) * np.power(12, 0.5)))
# %%
from sklearn.linear_model import RidgeCV

from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

loss, Y_test, Y_pre = [], [], []
weights = []
for year in range(10):
    x_train = x[x['y'] <= 2008 + year].iloc[:, 2:]
    y_train = y[y['y'] <= 2008 + year].iloc[:, -1]

    x_val = x[(x['y'] >= 2009 + year) & (x['y'] <= 2011 + year)].iloc[:, 2:]
    y_val = y[(y['y'] >= 2009 + year) & (y['y'] <= 2011 + year)].iloc[:, -1]

    x_test = x[x['y'] == 2012 + year].iloc[:, 2:]
    y_test = y[y['y'] == 2012 + year].iloc[:, -1]

    x_train_val = x[x['y'] <= 2011 + year].iloc[:, 2:]
    y_train_val = y[y['y'] <= 2011 + year].iloc[:, -1]

    test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
    test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
    ps = PredefinedSplit(test_fold=test_fold)
    param_grid = {'alpha':[0.1, 0.01, 0.001, 0.0001]}
    ENet = SGDRegressor(loss='huber',
                        penalty='l2',
                        # l1_ratio=0.5,
                        epsilon=1.35,
                        learning_rate='adaptive',
                        verbose=0)
    model = GridSearchCV(ENet, param_grid, cv = ps)
    history = model.fit(x_train_val, y_train_val,
                        )

    print('第{}年完成'.format(year + 1))

    pred_y3 = model.predict(x_test).reshape(len(y_test), 1)
    print('预测值为:', pred_y3)

    R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
        sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
    print('Ridge的R方：', R方_RF)

    pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/中国市场预测结果/AE/Ridge-{}.csv'.format(year + 1), index=False)

    # 因子重要性
    col = ['size', 'beta', 'mom', 'liq', 'vol', 'bpr', 'ey', 'growth', 'lever']
    for i in range(9):
        x2 = x_test.copy()
        feature = col[i]
        for j in range(11):
            x2.iloc[:, i + (j * 15)] = 0
        pd.DataFrame(
            model.predict(x2)).to_csv('/20230328代码与数据/中国市场预测结果/AE/Ridge-{}-{}.csv'.format(year + 1, feature),
                                      index=False)

    loss.append(history)
    Y_test.append(y_test)
    Y_pre.append(pred_y3)
    #weights.append(list(model.coef_))

data = pd.DataFrame()
pre = Y_pre[0]
for i in range(1, 10):
    pre = np.vstack([pre, Y_pre[i]])

data['pred'] = pre.reshape(len(pre), )
data['y'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'y'].values
data['m'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'm'].values
data['y_true'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'ret+1'].values

y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
R2_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
    sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
print('最终Ridge的R方：', R2_RF)


ret_long = []
ret_short = []
for year in range(10):
    for month in range(12):
        df = data[(data['y'] == 2012 + year) & (data['m'] == 1 + month)]
        df = df.sort_values('pred', ascending=False)
        ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), -1]))
        ret_short.append(np.mean(df.iloc[-int(len(df) / 10):, -1]))
        # print('第{}年:'.format(year + 1))
        # print('多:', ret_long[year * 12 + month], '空:', ret_short[year * 12 + month], '多空：',
        #       ret_long[year * 12 + month] - ret_short[year * 12 + month], '累计收益:', sum(ret_long) - sum(ret_short))
        # print('累计多:', sum(ret_long), '累计空:', sum(ret_short))
print('多空平均超额月收益率：', np.mean(np.array(ret_long) - np.array(ret_short)))
print('夏普比率:', np.mean(np.array(ret_long) - np.array(ret_short)) * 12 / (
            np.std(np.array(ret_long) - np.array(ret_short)) * np.power(12, 0.5)))
# %%
from sklearn.linear_model import ElasticNetCV

from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

loss, Y_test, Y_pre = [], [], []
weights = []
for year in range(10):
    x_train = x[x['y'] <= 2008 + year].iloc[:, 2:]
    y_train = y[y['y'] <= 2008 + year].iloc[:, -1]

    x_val = x[(x['y'] >= 2009 + year) & (x['y'] <= 2011 + year)].iloc[:, 2:]
    y_val = y[(y['y'] >= 2009 + year) & (y['y'] <= 2011 + year)].iloc[:, -1]

    x_test = x[x['y'] == 2012 + year].iloc[:, 2:]
    y_test = y[y['y'] == 2012 + year].iloc[:, -1]

    x_train_val = x[x['y'] <= 2011 + year].iloc[:, 2:]
    y_train_val = y[y['y'] <= 2011 + year].iloc[:, -1]

    test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
    test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
    ps = PredefinedSplit(test_fold=test_fold)
    param_grid = {'alpha':[0.001, 0.0001]}
    ENet = SGDRegressor(loss='huber',
                        penalty='elasticnet',
                        l1_ratio=0.5,
                        epsilon=1.35,
                        learning_rate='adaptive',
                        verbose=0)
    model = GridSearchCV(ENet, param_grid, cv = ps)
    history = model.fit(x_train_val, y_train_val,
                        )

    print('第{}年完成'.format(year + 1))

    pred_y3 = model.predict(x_test).reshape(len(y_test), 1)
    print('预测值为:', pred_y3)

    R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
        sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
    print('ENet的R方：', R方_RF)

    pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/中国市场预测结果/AE/ENet-{}.csv'.format(year + 1), index=False)

    # 因子重要性
    col = ['size', 'beta', 'mom', 'liq', 'vol', 'bpr', 'ey', 'growth', 'lever']
    for i in range(9):
        x2 = x_test.copy()
        feature = col[i]
        for j in range(11):
            x2.iloc[:, i + (j * 15)] = 0
        pd.DataFrame(
            model.predict(x2)).to_csv('/20230328代码与数据/中国市场预测结果/AE/ENet-{}-{}.csv'.format(year + 1, feature),
                                      index=False)

    loss.append(history)
    Y_test.append(y_test)
    Y_pre.append(pred_y3)
    #weights.append(list(model.coef_))

data = pd.DataFrame()
pre = Y_pre[0]
for i in range(1, 10):
    pre = np.vstack([pre, Y_pre[i]])

data['pred'] = pre.reshape(len(pre), )
data['y'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'y'].values
data['m'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'm'].values
data['y_true'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'ret+1'].values

y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
R2_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
    sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
print('最终ENet的R方：', R2_RF)



ret_long = []
ret_short = []
for year in range(10):
    for month in range(12):
        df = data[(data['y'] == 2012 + year) & (data['m'] == 1 + month)]
        df = df.sort_values('pred', ascending=False)
        ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), -1]))
        ret_short.append(np.mean(df.iloc[-int(len(df) / 10):, -1]))
        # print('第{}年:'.format(year + 1))
        # print('多:', ret_long[year * 12 + month], '空:', ret_short[year * 12 + month], '多空：',
        #       ret_long[year * 12 + month] - ret_short[year * 12 + month], '累计收益:', sum(ret_long) - sum(ret_short))
        # print('累计多:', sum(ret_long), '累计空:', sum(ret_short))
print('多空平均超额月收益率：', np.mean(np.array(ret_long) - np.array(ret_short)))
print('夏普比率:', np.mean(np.array(ret_long) - np.array(ret_short)) * 12 / (
            np.std(np.array(ret_long) - np.array(ret_short)) * np.power(12, 0.5)))
# %%
'''from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

loss, Y_test, Y_pre = [], [], []
weights = []
for year in range(10):
    x_train = x[x['y'] <= 2008 + year].iloc[:, 2:]
    y_train = y[y['y'] <= 2008 + year].iloc[:, -1]

    x_val = x[(x['y'] >= 2009 + year) & (x['y'] <= 2011 + year)].iloc[:, 2:]
    y_val = y[(y['y'] >= 2009 + year) & (y['y'] <= 2011 + year)].iloc[:, -1]

    x_test = x[x['y'] == 2012 + year].iloc[:, 2:]
    y_test = y[y['y'] == 2012 + year].iloc[:, -1]

    x_train_val = x[x['y'] <= 2011 + year].iloc[:, 2:]
    y_train_val = y[y['y'] <= 2011 + year].iloc[:, -1]

    test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
    test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
    ps = PredefinedSplit(test_fold=test_fold)

    param_grid = {'max_features': [3, 5, 10],
                  'max_depth': [1, 2],
                  'n_estimators': [50, 100, 150, 200, 300],
                  }
    model = RandomForestRegressor(n_jobs=15)
    model = GridSearchCV(model, param_grid, cv=ps, n_jobs=10,
                         )
    history = model.fit(x_train_val, y_train_val,
                        )

    print('第{}年完成'.format(year + 1))

    pred_y3 = model.predict(x_test).reshape(len(y_test), 1)
    print('预测值为:', pred_y3)

    R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
        sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
    print('RF的R方：', R方_RF)

    pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/中国市场预测结果/AE/RF-{}.csv'.format(year + 1), index=False)

    # 因子重要性
    col = ['size', 'beta', 'mom', 'liq', 'vol', 'bpr', 'ey', 'growth', 'lever']
    for i in range(9):
        x2 = x_test.copy()
        feature = col[i]
        for j in range(11):
            x2.iloc[:, i + (j * 15)] = 0
        pd.DataFrame(
            model.predict(x2)).to_csv('/20230328代码与数据/中国市场预测结果/AE/RF-{}-{}.csv'.format(year + 1, feature),
                                      index=False)

    loss.append(history)
    Y_test.append(y_test)
    Y_pre.append(pred_y3)

data = pd.DataFrame()
pre = Y_pre[0]
for i in range(1, 10):
    pre = np.vstack([pre, Y_pre[i]])

data['pred'] = pre.reshape(len(pre), )
data['y'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'y'].values
data['m'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'm'].values
data['y_true'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'ret+1'].values

y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
R2_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
    sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
print('最终RF的R方：', R2_RF)



ret_long = []
ret_short = []
for year in range(10):
    for month in range(12):
        df = data[(data['y'] == 2012 + year) & (data['m'] == 1 + month)]
        df = df.sort_values('pred', ascending=False)
        ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), -1]))
        ret_short.append(np.mean(df.iloc[-int(len(df) / 10):, -1]))
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


inpu = Input(int(x.shape[1]) - 2)


from sklearn.model_selection import PredefinedSplit
from sklearn.model_selection import GridSearchCV

loss,Y_test,Y_pre = [],[],[]
for year in range(10):
    x_train = x[x['y'] <= 2008 + year].iloc[:, 2:]
    y_train = y[y['y'] <= 2008 + year].iloc[:, -1]

    x_val = x[(x['y'] >= 2009 + year) & (x['y'] <= 2011 + year)].iloc[:, 2:]
    y_val = y[(y['y'] >= 2009 + year) & (y['y'] <= 2011 + year)].iloc[:, -1]

    x_test = x[x['y'] == 2012 + year].iloc[:, 2:]
    y_test = y[y['y'] == 2012 + year].iloc[:, -1]

    x_train_val = x[x['y'] <= 2011 + year].iloc[:, 2:]
    y_train_val = y[y['y'] <= 2011 + year].iloc[:, -1]

    test_fold = np.zeros(x_train_val.shape[0])  # 将所有index初始化为0,0表示第一轮的验证集
    test_fold[:x_train.shape[0]] = -1  # 将训练集对应的index设为-1，表示永远不划分到验证集中
    ps = PredefinedSplit(test_fold=test_fold)

    # gridsearch
    lr_list = [0.01, 0.001]  #
    l1_list = [0.001, 0.0001]  #
    batch_size_list = [2048, 10000]

    model_list = []
    score_list = []
    for lr in lr_list:
        for l1 in l1_list:
            for batch_size in batch_size_list:
                mini_model_list = [lr, l1, batch_size]
                pred_y3_list = []

                model = CNN(inpu=inpu, l1=l1, seed=0)
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
                score_list.append(1 - (sum((np.array(y_val).reshape(len(y_val), 1) - pred_y3) ** 2)) / (
                    sum((np.array(y_val).reshape(len(y_val), 1) ** 2))))
                model_list.append(mini_model_list)
                
    best_model_index = score_list.index(max(score_list))

    pred_y3_list = []
    final_model_list = []
    for ensemble in range(10):
        model = CNN(inpu=inpu, l1=model_list[best_model_index][1], seed=ensemble)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=model_list[best_model_index][0]),
                      loss='mse')
        model.fit(x_train, y_train,
                  batch_size=model_list[best_model_index][2],
                  epochs=100,
                  validation_data=(x_val, y_val),
                  callbacks=my_callbacks,
                  shuffle=True,
                  verbose=0
                  )
        pred_y3 = model.predict(x_test).reshape(-1, 1)

        final_model_list.append(model)
        pred_y3_list.append(pred_y3)
    pred_y3 = np.hstack(pred_y3_list)
    pred_y3 = np.mean(pred_y3, axis=1).reshape(-1, 1)

    R方_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pred_y3) ** 2)) / (
        sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
    print('NN3的R方：', R方_RF)

    pd.DataFrame(pred_y3).to_csv('/20230328代码与数据/中国市场预测结果/AE/NN3-{}.csv'.format(year + 1), index=False)

    # 因子重要性
    col = ['size', 'beta', 'mom', 'liq', 'vol', 'bpr', 'ey', 'growth', 'lever']
    for i in range(9):
        x2 = x_test.copy()
        feature = col[i]
        for j in range(11):
            x2.iloc[:, i + (j * 15)] = 0
        pred_list = []
        for model in final_model_list:
            pred_list.append(model.predict(x2).reshape(-1, 1))
        pred = np.hstack(pred_list)
        pred = np.mean(pred, axis=1).reshape(-1, 1)
        pd.DataFrame(pred).to_csv('/20230328代码与数据/中国市场预测结果/AE/NN3-{}-{}.csv'.format(year + 1, feature), index=False)

    Y_test.append(y_test)
    Y_pre.append(pred_y3)

data = pd.DataFrame()
pre = Y_pre[0]
for i in range(1, 10):
    pre = np.vstack([pre, Y_pre[i]])

data['pred'] = pre.reshape(len(pre), )
data['y'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'y'].values
data['m'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'm'].values
data['y_true'] = y[(y['y'] >= 2012) & (y['y'] <= 2021)].loc[:, 'ret+1'].values

y_test = np.array(data['y_true']).reshape(len(data['y_true']), 1)
R2_RF = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - pre) ** 2)) / (
    sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))
print('最终NN3的R方：', R2_RF)



ret_long = []
ret_short = []
for year in range(10):
    for month in range(12):
        df = data[(data['y'] == 2012 + year) & (data['m'] == 1 + month)]
        df = df.sort_values('pred', ascending=False)
        ret_long.append(np.mean(df.iloc[0:int(len(df) / 10), -1]))
        ret_short.append(np.mean(df.iloc[-int(len(df) / 10):, -1]))
        # print('第{}年:'.format(year + 1))
        # print('多:', ret_long[year * 12 + month], '空:', ret_short[year * 12 + month], '多空：',
        #       ret_long[year * 12 + month] - ret_short[year * 12 + month], '累计收益:', sum(ret_long) - sum(ret_short))
        # print('累计多:', sum(ret_long), '累计空:', sum(ret_short))
print('多空平均超额月收益率：', np.mean(np.array(ret_long) - np.array(ret_short)))
print('夏普比率:', np.mean(np.array(ret_long) - np.array(ret_short)) * 12 / (
            np.std(np.array(ret_long) - np.array(ret_short)) * np.power(12, 0.5)))'''