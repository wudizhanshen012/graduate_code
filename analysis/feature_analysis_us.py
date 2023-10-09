import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

from paper_final_project.utils.feature_analysis import FeatureAnalysis

def R2(y_true, y_pre):
    return 1 - (np.sum((np.array(y_true).reshape(len(y_true), 1) - y_pre) ** 2)) / (np.sum((np.array(y_true).reshape(len(y_true), 1) ** 2)))

FA = FeatureAnalysis()

# 读取数据
data = pd.read_csv("/20230328代码与数据/美国市场数据/final_data——1965-2020.csv", encoding='utf-8')
data['ym'] = data['DATE'].astype(str).apply(lambda x: x[0:4]+'-'+x[4:6])
data = data.set_index(data['ym'])
data.index = pd.to_datetime(data.index)
del data['DATE']
del data['permno']
del data['ym']

data = data.rename(columns={'R-1': 'ret+1'})

# performance analysis
time_period = data['2001':'2019'].index
y_true = data['2001':'2019'].loc[:, ['ret+1']].values
size = data['2001':'2019'].loc[:, ['mvel1']].values

path = "/20230328代码与数据/美国市场预测结果/AE"
model_list = ['OLS', 'Lasso', 'Ridge', 'ENet', 'RF', 'NN3']
RET, STD, SR, MAX_DD, Cumsum_Ret = [], [], [], [], []
LONG_RET, LONG_STD, LONG_SR, LONG_MAX_DD, LONG_Cumsum_Ret = [], [], [], [], []
for model1 in model_list:
    y_pre1_list = []
    for year in range(1, 20):
        y_pre1 = pd.read_csv(path + '/' + model1 + '-' + str(year) + '.csv')
        print(path + '/' + model1 + '-' + str(year) + '.csv')
        y_pre1 = np.array(y_pre1).reshape(-1, 1)
        y_pre1_list.append(pd.DataFrame(y_pre1))
    y_pre1 = pd.concat(y_pre1_list).values
    mean_ret, std, Sharpe_Ratio, Max_dd, cumsum_ret, \
    long_mean_ret, long_std, long_Sharpe_Ratio, long_Max_dd, long_cumsum_ret = FA.performance_analysis(time_period,
                                                                                          y_true,
                                                                                          y_pre1,
                                                                                          size,
                                                                                          is_value=True)
    RET.append(mean_ret)
    STD.append(std)
    SR.append(Sharpe_Ratio)
    MAX_DD.append(Max_dd)
    Cumsum_Ret.append(cumsum_ret)
    LONG_RET.append(long_mean_ret)
    LONG_STD.append(long_std)
    LONG_SR.append(long_Sharpe_Ratio)
    LONG_MAX_DD.append(long_Max_dd)
    LONG_Cumsum_Ret.append(long_cumsum_ret)

performance_df = pd.DataFrame()
performance_df['ret'] = RET
performance_df['std'] = STD
performance_df['Max_dd'] = MAX_DD
performance_df['Sharpe_Ratio'] = SR
performance_df.index = model_list
performance_df = performance_df.T
performance_df.to_csv('/cta-model-template/paper_final_project/us_ae/value_performance_ls_us.csv')
# only long
performance_df = pd.DataFrame()
performance_df['ret'] = LONG_RET
performance_df['std'] = LONG_STD
performance_df['Max_dd'] = LONG_MAX_DD
performance_df['Sharpe_Ratio'] = LONG_SR
performance_df.index = model_list
performance_df = performance_df.T
performance_df.to_csv('/cta-model-template/paper_final_project/us_ae/value_performance_long_only_us.csv')

for i in range(6):
    plt.plot(Cumsum_Ret[i], label=model_list[i])
plt.legend(loc='best')
plt.savefig('/cta-model-template/paper_final_project/us_ae/value_performance_ls_us.png')
plt.show()
for i in range(6):
    plt.plot(LONG_Cumsum_Ret[i], label=model_list[i])
plt.legend(loc='best')
plt.savefig('/cta-model-template/paper_final_project/us_ae/value_performance_long_only_us.png')
plt.show()

raise TypeError('done')

# 计算美国市场R2重要性
# y_true = data['2001':'2020'].loc[:, ['ret+1']]
# AE_data = pd.read_csv('E:/代码与数据/AE_data_us_202305.csv')
# AE_data.index = data.index
# AE_feature = AE_data['2001':'2020']

# x_y = pd.concat([AE_feature, y_true], axis=1).reset_index()
# for i in ['ey', 'growth', 'lever', 'price', 'volume']:
#     group_ret, final = FA.single_factor_analysis(x_y.loc[:, [i, 'ret+1', 'ym']],
#                                                  size=data['2001':'2020'].loc[:, ['mvel1']], start_year=2001,
#                                                  end_year=2020, end_month=12, is_value=True)
#     final.to_csv(f'E:/代码与数据/xiu最新代码/final_project/single_factor_analysis_performance/value_{i}_performance.csv')
#
# path = "F:/xiu预测结果/AE202304"
# filelist = os.listdir(path)
# R2_array = []  # 每行表示每个特征的重要性，每列表示每个模型
# for feature in ['ey', 'growth', 'lever', 'price', 'volume']:
#     feature_R2_list = []
#     for model in ['OLS', 'Lasso', 'Ridge', 'ENet', 'RF', 'NN3']:
#         if model != 'NN3':
#             path1 = path
#         else:
#             path1 = path
#         print(f'{model}-{path1}')
#         y_pre_list = []
#         for year in range(1, 21):
#             y_pre = pd.read_csv(path1 + '/' + model + '_new-' + str(year) + '-' + feature + '.csv')
#             y_pre = np.array(y_pre).reshape(-1, 1)
#             y_pre_list.append(pd.DataFrame(y_pre))
#         y_pre = pd.concat(y_pre_list)
#         feature_R2_list.append(R2(y_true=y_true,
#                                   y_pre=y_pre))
#     R2_array.append(feature_R2_list)
# R2_array = np.array(R2_array).reshape(-1, 6)
#
# # 让各个模型减去不去掉特征的R方，计算该因子的重要性
# # ols: 0.02, lasso: 0.56, ridge: 0.02, elasticnet: 0.56, RF: 0.11, NN3: 0.49
# origin_R2_score = [0.0002, 0.0056, 0.0002, 0.0056, 0.0011, 0.0049]
# for i in range(len(origin_R2_score)):
#     R2_array[:, i] = origin_R2_score[i] - R2_array[:, i]
#
# R2_array = StandardScaler().fit_transform(R2_array)
# R2_array = pd.DataFrame(R2_array)
# R2_array.columns = ['OLS', 'Lasso', 'Ridge', 'ENet', 'RF', 'NN3']
# R2_array.index = ['ey', 'growth', 'lever', 'price', 'volume']
# R2_array['sort'] = R2_array.sum(axis=1).values
# R2_array = R2_array.sort_values('sort', ascending=False).iloc[:, :-1]
# model_name = list(R2_array.columns)
# feature_name = list(R2_array.index)
#
# R2_array = np.array(R2_array)
# FA.feature_importance(R2_array,
#                       model_name=model_name,
#                       feature=feature_name)

# DM检验
y_true = data['2001':'2020'].loc[:, ['ret+1']].values.reshape(-1).tolist()

path1 = "F:/xiu预测结果/AE202304"
path2 = "F:/xiu预测结果/原始202304"
model_list = ['OLS', 'Lasso', 'Ridge', 'ENet', 'RF', 'NN3']
for model in model_list:
    if model == 'NN3':
        path1 = path1
    y_pre1_list = []
    y_pre2_list = []
    for year in range(1, 21):
        y_pre1 = pd.read_csv(path1 + '/' + model + '_new-' + str(year) + '.csv')
        y_pre1 = np.array(y_pre1).reshape(-1, 1)
        y_pre1_list.append(pd.DataFrame(y_pre1))

        y_pre2 = pd.read_csv(path2 + '/' + model + '-' + str(year) + '.csv')
        y_pre2 = np.array(y_pre2).reshape(-1, 1)
        y_pre2_list.append(pd.DataFrame(y_pre2))
    y_pre1 = pd.concat(y_pre1_list).values.reshape(-1).tolist()
    y_pre2 = pd.concat(y_pre2_list).values.reshape(-1).tolist()

    print(f'{model}的dm检验结果为：', FA.NW_DM_test(y_pre1, y_pre2, y_true, data['2001':'2020'].index))

