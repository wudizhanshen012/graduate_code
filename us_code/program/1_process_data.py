# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import warnings
from sklearn import preprocessing
import datetime
from us_code.config import *

from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Input
from tensorflow.keras.layers import Dropout

warnings.filterwarnings('ignore')

MinMax = preprocessing.MinMaxScaler(feature_range=(-1, 1))

# base data
data = pd.read_csv(data_path + "origin_data/final_data——1965-2020.csv", encoding='utf-8')
data['ym'] = data['DATE'].astype(str).apply(lambda x: x[0:4] + '-' + x[4:6])
data['ym'] = pd.to_datetime(data['ym'])
del data['DATE']
data = data.rename(columns={'R-1': 'ret+1'})
data = data[data['ym'] >= '1965-01']
ret_data = data[['permno', 'ym', 'ret+1']]

a = pd.read_csv(data_path + "origin_data/datashare.csv", encoding='utf-8')
a['ym'] = a['DATE'].astype(str).apply(lambda x: x[0:4] + '-' + x[4:6])
a['ym'] = pd.to_datetime(a['ym'])
a = pd.merge(a, ret_data, on=['permno', 'ym'], how='left')
data = a.copy()
data = data[(data['ym'] >= '1965-01') & (data['ym'] <= '2020-12')]
del a
del ret_data

# macro
df = pd.read_csv(data_path + "origin_data/PredictorData2021.xlsx-Monthly.csv")
df['Dp'] = np.log(df['D12'] / df['Index'])  # d/p
df['Ep'] = np.log(df['E12'] / df['Index'])  # e/p
df['Bm'] = df['b/m']  # B/M
df['Ntis'] = df['ntis']  # net equity expansion
df['Tbl'] = df['tbl']  # Treasury-bill rate
df['Tms'] = df['lty'] - df['tbl']  # term-spread
df['Dfy'] = df['BAA'] - df['AAA']  # default-spread
df['Svar'] = df['svar']  # stock variance 或许不需要差分 todo
df['ym'] = df['yyyymm'].astype(str).apply(lambda x: x[0:4] + '-' + x[4:6])
df['ym'] = pd.to_datetime(df['ym'])
df['Rf+1'] = df['tbl'].shift(-1)
df['Rf+1'] /= 12
df = df[['ym', 'Dp', 'Ep', 'Bm', 'Ntis', 'Tbl', 'Tms', 'Dfy', 'Svar', 'Rf+1']]

# 宏观数据差分+时序标准化
df.iloc[:, 1:-1] = df.iloc[:, 1:-1].pct_change()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.fillna(method='ffill')
df = df.fillna(0)
#
df = df.set_index(['ym', 'Rf+1'])
max_data = df.rolling(window=72, min_periods=72).max()
min_data = df.rolling(window=72, min_periods=72).min()
df = (df - min_data) / (max_data - min_data)
df = df.reset_index()

# merge stock data and macro data
data = pd.merge(data, df, on=['ym'], how='left')

# 个股数据截面标准化和计算超额收益率

# 对data按照ym列进行分组，并对每组中的x_col列表中的元素进行标准化
data_list = []
for key, value in data.groupby('ym'):
    value[x_col] = MinMax.fit_transform(value[x_col])
    data_list.append(value)
data = pd.concat(data_list)
del data_list
data[y_col[0]] -= data['Rf+1']

data = data.fillna(0)
assert data.shape == data.dropna().shape

data.to_pickle(data_path + 'processed_data/origin_data.pkl')
print('origin done')

# PCA
# train_data = data[data['ym'] <= '2000-12-31']
# for category in category_factor_dict:
#     pca = PCA(n_components=1)
#     pca.fit(train_data[category_factor_dict[category]])
#     data['pca_' + category] = pca.transform(data[category_factor_dict[category]])
#
# pca_data = data[['permno', 'ym', 'sic2', 'ret+1'] + ['sin', 'baspread'] + list(map(lambda s: 'pca_' + s, list(category_factor_dict.keys()))) + macro_col]
# pca_data.to_pickle(data_path + 'processed_data/pca_data.pkl')
# del pca_data
# print('pca done')

# PLS
# for category in category_factor_dict:
#     pls = PLSRegression(n_components=1)
#     pls.fit(X = train_data[category_factor_dict[category]], Y = train_data[y_col[0]])
#     data['pls_' + category] = pls.transform(data[category_factor_dict[category]])
#
# pls_data = data[['permno', 'ym', 'sic2', 'ret+1'] + ['sin', 'baspread'] + list(map(lambda s: 'pls_' + s, list(category_factor_dict.keys()))) + macro_col]
# pls_data.to_pickle(data_path + 'processed_data/pls_data.pkl')
# del pls_data
# print('pls done')

# data = pd.read_pickle(data_path + 'processed_data/origin_data.pkl')

for category in category_factor_dict:
    temp = pd.read_csv(data_path + f'/processed_data/ae_{category}.csv')
    data['ae_' + category] = temp['ae_' + category]
    print(category, 'done!')

AE_data = data[['permno', 'ym', 'sic2', 'ret+1'] + ['sin', 'baspread'] + list(map(lambda s: 'ae_' + s, list(category_factor_dict.keys()))) + macro_col]
AE_data.to_pickle(data_path + 'processed_data/ae_data.pkl')

# # AE
# my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
#
# train_data = data[(data['ym'] >= '1965-01') & (data['ym'] <= '1986-12')]
# val_data = data[(data['ym'] >= '1987-01') & (data['ym'] <= '2000-12')]
# test_data = data[(data['ym'] >= '2001-01') & (data['ym'] <= '2020-12')]
#
# for category in category_factor_dict:
#
#     x_train = train_data[category_factor_dict[category]]
#     x_val = val_data[category_factor_dict[category]]
#
#     M = []
#     M_final = []
#     lr_list = [0.001, 0.0001]
#     neuron_list = [4, 16, 64]
#     dropout_list = [0, 0.2, 0.5]
#     for lr in lr_list:
#         for neuron in neuron_list:
#             for dropout in dropout_list:
#                 # in_cell = 5
#                 input_dim = np.array(x_train).shape[1]
#                 latent_dim = 1
#                 batch_size = 10000
#                 my_callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)]
#
#                 inputs = Input(shape=(input_dim))
#
#                 encoded = Dense(neuron, activation='tanh')(inputs)
#                 encoded = BatchNormalization()(encoded)
#                 encoded = Dropout(dropout)(encoded)
#                 encoded = Dense(latent_dim, activation='tanh')(encoded)
#
#                 # dncoded = RepeatVector(timesteps)(encoded)
#                 decoded = BatchNormalization()(encoded)
#                 decoded = Dense(neuron, activation='tanh')(decoded)
#                 decoded = BatchNormalization()(decoded)
#                 decoded = Dropout(dropout)(decoded)
#                 decoded = Dense(input_dim, activation='tanh')(decoded)
#
#                 encoder = Model(inputs, encoded)
#                 autoencoder = Model(inputs, decoded)
#                 autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')  #
#
#                 autoencoder.fit(x_train, x_train, epochs=100, batch_size=batch_size,
#                                 validation_data=(x_val, x_val),
#                                 callbacks=my_callbacks,
#                                 verbose=0
#                                 )
#
#                 M.append(autoencoder)
#                 M_final.append(encoder)
#     val_mse = []
#     for i in M:
#         pred = i.predict(x_val)
#         val = np.array(x_val)
#         mse = ((pred - val) ** 2).mean()
#         val_mse.append(mse)
#     position = val_mse.index(min(val_mse))
#     encoder = M_final[position]
#
#     data['ae_' + category] = encoder.predict(data[category_factor_dict[category]]).reshape(len(data), )
#     data['ae_' + category].to_csv(data_path + 'processed_data/ae_' + category + '.csv')
#     print(category, 'done!')
#
# AE_data = data[['permno', 'ym', 'sic2', 'ret+1'] + ['sin', 'baspread'] + list(map(lambda s: 'ae_' + s, list(category_factor_dict.keys()))) + macro_col]
# AE_data.to_pickle(data_path + 'processed_data/AE_data.pkl')