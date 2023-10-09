import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

from final_project.utils.feature_analysis import FeatureAnalysis

def R2(y_true, y_pre):
    return 1 - (np.sum((np.array(y_true).reshape(len(y_true), 1) - y_pre) ** 2)) / (np.sum((np.array(y_true).reshape(len(y_true), 1) ** 2)))

FA = FeatureAnalysis()

data = pd.read_csv("E:/代码与数据/final_data_china_2022-12-22.csv",encoding='utf-8')
data = data.set_index(data['Accper'])
data.index = pd.to_datetime(data.index)
data = data['2000': '2022']
del data['Stkcd']

data = data.rename(columns={'R-1': 'ret+1'})

AE_data = pd.read_csv('E:/代码与数据/AE_data_china_202304.csv')
AE_data.index = data.index
AE_data['ym'] = data['Accper']
AE_data['ret+1'] = data['ret+1']

factor_list = ['ey', 'growth', 'lever', 'price', 'volume']
macro_list = ['dp', 'de', 'bm_macro', 'ep', 'ntis', 'svar',
       'tms', 'mtr', 'infl', 'm2gr', 'itgr']
AE_data['Intercept'] = 1
# for f in factor_list:
#     result = FA.fama_macbeth(AE_data, 'ym', 'ret+1', [f, 'Intercept']+macro_list)
#     final = FA.get_summary(result)
#     print(final)
# result = FA.fama_macbeth(AE_data, 'ym', 'ret+1', factor_list+macro_list+['Intercept'])
# final = FA.get_summary(result)
# print(final)


# performance analysis
time_period = data['2012':'2021'].index
y_true = data['2012':'2021'].loc[:, ['ret+1']].values
size = data['2012':'2021'].loc[:, ['mve']].values

path = "F:/中国市场预测结果/原始202304"
model_list = ['OLS', 'Lasso', 'Ridge', 'ENet', 'RF', 'NN3']
RET, STD, SR, MAX_DD, Cumsum_Ret = [], [], [], [], []
LONG_RET, LONG_STD, LONG_SR, LONG_MAX_DD, LONG_Cumsum_Ret = [], [], [], [], []
for model1 in model_list:
    y_pre1_list = []
    for year in range(1, 11):
        y_pre1 = pd.read_csv(path + '/' + model1 + '-' + str(year) + '.csv')
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
performance_df.to_csv('value_performance_ls_china.csv')
# only long
performance_df = pd.DataFrame()
performance_df['ret'] = LONG_RET
performance_df['std'] = LONG_STD
performance_df['Max_dd'] = LONG_MAX_DD
performance_df['Sharpe_Ratio'] = LONG_SR
performance_df.index = model_list
performance_df = performance_df.T
performance_df.to_csv('value_performance_long_only_china.csv')

for i in range(6):
    plt.plot(Cumsum_Ret[i], label=model_list[i])
plt.legend(loc='best')
plt.savefig('value_performance_ls_china.png')
plt.show()
for i in range(6):
    plt.plot(LONG_Cumsum_Ret[i], label=model_list[i])
plt.legend(loc='best')
plt.savefig('value_performance_long_only_china.png')
plt.show()
#
# # 计算市场R2重要性
# y_true = data['2012':'2021'].loc[:, ['ret+1']]
# AE_data = pd.read_csv('E:/代码与数据/AE_data_china_202304.csv')
# AE_data.index = data.index
# AE_data['ym'] = data['Accper']
# AE_feature = AE_data['2012':'2021']
#
# x_y = pd.concat([AE_feature, y_true], axis=1).reset_index()
# for i in ['size_', 'beta_', 'mom_', 'liq_', 'vol_', 'bpr_', 'ey_', 'growth_', 'lever_']:
#     group_ret, final = FA.single_factor_analysis(x_y.loc[:, [i, 'ret+1', 'ym']],
#                                                  size=data['2012':'2021'].loc[:, ['mve']], start_year=2012,
#                                                  end_year=2021, end_month=13,
#                                                  is_value=False)
#     final.to_csv(f'E:/代码与数据/xiu最新代码/final_project/single_factor_analysis_performance/avg_{i}_performance_china.csv')

# path = "F:/中国市场预测结果/AE202304"
# filelist = os.listdir(path)
# R2_array = []  # 每行表示每个特征的重要性，每列表示每个模型
# for feature in factor_list:
#     feature_R2_list = []
#     for model in ['OLS', 'Lasso', 'Ridge', 'ENet', 'RF', 'NN3']:
#         y_pre_list = []
#         for year in range(1, 11):
#             y_pre = pd.read_csv(path + '/' + model + '-' + str(year) + '-' + feature + '.csv')
#             y_pre = np.array(y_pre).reshape(-1, 1)
#             y_pre_list.append(pd.DataFrame(y_pre))
#         y_pre = pd.concat(y_pre_list)
#         feature_R2_list.append(R2(y_true=y_true,
#                                   y_pre=y_pre))
#     R2_array.append(feature_R2_list)
# R2_array = np.array(R2_array).reshape(9, 6)
#
# # 让各个模型减去不去掉特征的R方，计算该因子的重要性
# origin_R2_score = [0.004, 0.005, 0.004, 0.0046, -0.0032, 0.12]
# for i in range(len(origin_R2_score)):
#     R2_array[:, i] = origin_R2_score[i] - R2_array[:, i]

#R2_array = StandardScaler().fit_transform(R2_array)
# R2_array = pd.DataFrame(R2_array)
# R2_array.columns = ['OLS', 'Lasso', 'Ridge', 'ENet', 'RF', 'NN3']
# R2_array.index = factor_list
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
y_true = data['2012':'2021'].loc[:, ['ret+1']].values.reshape(-1).tolist()

path1 = "F:/中国市场预测结果/AE202304"
path2 = "F:/中国市场预测结果/原始202304"
model_list = ['OLS', 'Lasso', 'Ridge', 'ENet', 'RF', 'NN3']
for model in model_list:
    y_pre1_list = []
    y_pre2_list = []
    for year in range(1, 11):
        y_pre1 = pd.read_csv(path1 + '/' + model + '-' + str(year) + '.csv')
        y_pre1 = np.array(y_pre1).reshape(-1, 1)
        y_pre1_list.append(pd.DataFrame(y_pre1))

        y_pre2 = pd.read_csv(path2 + '/' + model + '-' + str(year) + '.csv')
        y_pre2 = np.array(y_pre2).reshape(-1, 1)
        y_pre2_list.append(pd.DataFrame(y_pre2))
    y_pre1 = pd.concat(y_pre1_list).values.reshape(-1).tolist()
    y_pre2 = pd.concat(y_pre2_list).values.reshape(-1).tolist()

    print(f'{model}与的dm检验结果为：', FA.NW_DM_test(y_pre1, y_pre2, y_true, data['2012':'2021'].index))
