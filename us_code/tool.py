import tensorflow as tf
import pandas as pd
import numpy as np

def init_cuda():
    '''
    让程序只使用要用的gpu内存，而不是全部使用
    '''
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    if len(gpus) != 0:
        tf.config.experimental.set_visible_devices(devices=gpus[0], device_type='GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

def industry_dummy(industry_codes):
    '''
    计算行业虚拟变量
    '''
    industry_df = pd.DataFrame(industry_codes, columns=['industry_code'])
    industry_df = pd.get_dummies(industry_df, columns=['industry_code'])
    return industry_df.values

def cal_R2(y_test, pred_y):
    R2 = 1 - (sum((np.array(y_test).reshape(len(y_test), 1) - np.array(pred_y).reshape(len(pred_y), 1)) ** 2)) / (
        sum((np.array(y_test).reshape(len(y_test), 1) ** 2)))

    return R2