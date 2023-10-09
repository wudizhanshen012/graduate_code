import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime

from collections import namedtuple
from scipy.stats import distributions

class FeatureAnalysis():
    '''FM回归
       首先使用fama_macbeth（总数据，日期，收益率名字，特征名字）
       将得到的结果放入get_summary就可以得到系数和t值'''

    def __init__(self, ):
        pass

    '''--------------------------------------------------------------------------------------------------
    以下是Fama-MacBeth回归代码'''

    def __np_ols(self, data, yvar, xvar):
        """
        Wrapper of np.linalg.lstsq
        Notes
        -----
        Under the hood, pseudoinverse is calculated using singular value decomposition (SVD), As any matrix can be decomposite as $A=U \Sigma V^T$, then pseudoinverse of matrix $A$ is $A^+ = V \Sigma^+ U^T$. `rcond` is used to set cut-off ratio for small singular values of in $\Sigma$. Setting `rcond=None` to silence the warning and use machine prcision as rcond parameter.
        [What does the rcond parameter of numpy.linalg.pinv do?](https://stackoverflow.com/questions/53949202/what-does-the-rcond-parameter-of-numpy-linalg-pinv-do)
        """

        betas, _, _, _ = np.linalg.lstsq(data[xvar], data[yvar], rcond=None)

        return pd.Series(betas.flatten())

    def hac_maxlags(self, t):
        """
        Calculate `maxlags` for Heteroskedasticity and Autocorrelation Consistent (HAC) estimator or Newey–West estimator
        ..math::
            J = [4 \\times T/100 ]^{2/9}
        Parameters
        ----------
        t : int
            length of time series
        Returns
        -------
        J : float
            maxlags

        Reference
        ---------
        [1] Newey, Whitney K., and Kenneth D. West, 1987, A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix, Econometrica 55, 703–708.
        """

        return np.floor(4 * (t / 100) ** (2 / 9))

    def from_formula(self, reg_formula: str):
        """
        Breaking list of regression formula into y and x variables
        Parameters
        ----------
        reg_formula : string
            string of regression formula
        Returns
        -------
        (yvar, xvar_list) : tuple
        yvar : string
            string of y variable name
        xvar_list : list
            list of x variable names

        Example
        -------
        >>> from_formula('y ~ intercept + x1')
        ('y', ['intercept', 'x1'])
        >>> from_formula('ret~1+x1+x2')
        ('ret', ['1', 'x1', 'x2', 'x3'])
        """

        yvar = reg_formula.split('~')[0].strip()
        x_string = reg_formula.split('~')[1].strip()
        xvar_list = [x.strip() for x in x_string.split('+')]

        return yvar, xvar_list

    def fama_macbeth(self, data, t, yvar, xvar):
        """
        Fama-macbeth regression for every t, `excess return ~ beta' lambda`, regressing ret on beta to get time series of lambdas (factor risk premium)
        Parameters
        ----------
        data : pd.DataFrame
            dataframe contains excess return and factor beta (exposure)
        t : string
            column name of date/time/periods
        yvar : string
            excess return of test asset
        xvar : list of strings
            factor betas
        Returns
        -------
        pd.DataFrame, shape of (len(t),len(xvar))
            return time series result of lambdas (factor risk premium)

        Notes
        -----
        If intercept is needed, add to xvar list.
        Reference
        ---------
        [1] Fama, Eugene F., and James D. MacBeth, 1973, Risk, Return, and Equilibrium: Empirical Tests, Journal of Political Economy 81, 607–636.
        """

        # running cross-sectional ols for every t, get time series lambdas
        d = (data.groupby(t).apply(self.__np_ols, yvar, xvar))
        # rename column names
        d.columns = xvar

        return d

    def get_summary(self, lambd, HAC=False, **kwargs):
        """_summary_
        Parameters
        ----------
        lambd : _type_
            _description_
        HAC : bool, optional
            _description_, by default False
        Returns
        -------
        _type_
            _description_
        """

        import statsmodels.formula.api as smf
        from scipy import stats

        s = lambd.describe().T
        # getting robust HAC estimators
        if HAC:
            if ('maxlags' in kwargs):
                maxlags = kwargs['maxlags']
                full_xvars = lambd.columns.to_list()
                std_error = []
                for var in full_xvars:
                    # calculate individual Newey-West adjusted standard error using `smf.ols`
                    reg = smf.ols('{} ~ 1'.format(var), data=lambd).fit(cov_type='HAC', cov_kwds={'maxlags': maxlags})
                    std_error.append(reg.bse[0])
                s['std_error'] = std_error
            else:
                print('`maxlag` is needed to computer HAC')
        else:
            # nonrobust estimators
            s['std_error'] = s['std'] / np.sqrt(s['count'])

        # t-statistics
        s['tstat'] = s['mean'] / s['std_error']
        # 2-sided p-value for the t-statistic
        s['pval'] = stats.t.sf(np.abs(s['tstat']), s['count'] - 1) * 2

        return s

    '''Fama-MacBeth结束
    -------------------------------------------------------------------------------------------------------'''

    '''特征下降R方重要性图绘画'''
    def feature_importance(self, data, model_name, feature):
        '''data:array,每行表示每个特征的重要性，每列表示每个模型
           model_name:list,模型名称
           feature:list,特征名称'''

        # R方数组，

        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(data, cmap=plt.get_cmap('Greens'))

        # 设置x,y轴刻度
        tick1 = np.arange(0, len(model_name), 1)
        tick2 = np.arange(0, len(feature), 1)
        ax.set_yticks(tick2)
        ax.set_xticks(tick1)

        # model_name = ['ols', 'lasso', 'NN']
        # feature = ['size', 'beta', 'vol']
        ax.set_yticklabels(feature)
        ax.set_xticklabels(model_name)
        plt.show()

    '''DM检验
       输入：model1 prediction, model2 prediction, original return'''

    def dm_test(self, actual_lst, pred1_lst, pred2_lst, h=1, crit="MSE", power=2):
        # Routine for checking errors
        def error_check():
            rt = 0
            msg = ""
            # Check if h is an integer
            if (not isinstance(h, int)):
                rt = -1
                msg = "The type of the number of steps ahead (h) is not an integer."
                return (rt, msg)
            # Check the range of h
            if (h < 1):
                rt = -1
                msg = "The number of steps ahead (h) is not large enough."
                return (rt, msg)
            len_act = len(actual_lst)
            len_p1 = len(pred1_lst)
            len_p2 = len(pred2_lst)
            # Check if lengths of actual values and predicted values are equal
            if (len_act != len_p1 or len_p1 != len_p2 or len_act != len_p2):
                rt = -1
                msg = "Lengths of actual_lst, pred1_lst and pred2_lst do not match."
                return (rt, msg)
            # Check range of h
            if (h >= len_act):
                rt = -1
                msg = "The number of steps ahead is too large."
                return (rt, msg)
            # Check if criterion supported
            if (crit != "MSE" and crit != "MAPE" and crit != "MAD" and crit != "poly"):
                rt = -1
                msg = "The criterion is not supported."
                return (rt, msg)
                # Check if every value of the input lists are numerical values
            from re import compile as re_compile
            comp = re_compile("^\d+?\.\d+?$")

            def compiled_regex(s):
                """ Returns True is string is a number. """
                if comp.match(s) is None:
                    return s.isdigit()
                return True

            for actual, pred1, pred2 in zip(actual_lst, pred1_lst, pred2_lst):
                is_actual_ok = compiled_regex(str(abs(actual)))
                is_pred1_ok = compiled_regex(str(abs(pred1)))
                is_pred2_ok = compiled_regex(str(abs(pred2)))
                if (not (is_actual_ok and is_pred1_ok and is_pred2_ok)):
                    msg = "An element in the actual_lst, pred1_lst or pred2_lst is not numeric."
                    rt = -1
                    return (rt, msg)
            return (rt, msg)

        # Error check
        error_code = error_check()
        # Raise error if cannot pass error check
        if (error_code[0] == -1):
            # raise SyntaxError(error_code[1])
            # return
            pass
        # Import libraries
        from scipy.stats import t
        import collections
        import pandas as pd
        import numpy as np

        # Initialise lists
        e1_lst = []
        e2_lst = []
        d_lst = []

        # convert every value of the lists into real values
        actual_lst = pd.Series(actual_lst).apply(lambda x: float(x)).tolist()
        pred1_lst = pd.Series(pred1_lst).apply(lambda x: float(x)).tolist()
        pred2_lst = pd.Series(pred2_lst).apply(lambda x: float(x)).tolist()

        # Length of lists (as real numbers)
        T = float(len(actual_lst))

        # construct d according to crit
        if (crit == "MSE"):
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                e1_lst.append((actual - p1) ** 2)
                e2_lst.append((actual - p2) ** 2)
            for e1, e2 in zip(e1_lst, e2_lst):
                d_lst.append(e1 - e2)
        elif (crit == "MAD"):
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                e1_lst.append(abs(actual - p1))
                e2_lst.append(abs(actual - p2))
            for e1, e2 in zip(e1_lst, e2_lst):
                d_lst.append(e1 - e2)
        elif (crit == "MAPE"):
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                e1_lst.append(abs((actual - p1) / actual))
                e2_lst.append(abs((actual - p2) / actual))
            for e1, e2 in zip(e1_lst, e2_lst):
                d_lst.append(e1 - e2)
        elif (crit == "poly"):
            for actual, p1, p2 in zip(actual_lst, pred1_lst, pred2_lst):
                e1_lst.append(((actual - p1)) ** (power))
                e2_lst.append(((actual - p2)) ** (power))
            for e1, e2 in zip(e1_lst, e2_lst):
                d_lst.append(e1 - e2)

                # Mean of d
        mean_d = pd.Series(d_lst).mean()

        # Find autocovariance and construct DM test statistics
        def autocovariance(Xi, N, k, Xs):
            autoCov = 0
            T = float(N)
            for i in np.arange(0, N - k):
                autoCov += ((Xi[i + k]) - Xs) * (Xi[i] - Xs)
            return (1 / (T)) * autoCov

        gamma = []
        for lag in range(0, h):
            gamma.append(autocovariance(d_lst, len(d_lst), lag, mean_d))  # 0, 1, 2
        V_d = (gamma[0] + 2 * sum(gamma[1:])) / T
        DM_stat = V_d ** (-0.5) * mean_d
        harvey_adj = ((T + 1 - 2 * h + h * (h - 1) / T) / T) ** (0.5)
        DM_stat = harvey_adj * DM_stat
        # Find p-value
        p_value = 2 * t.cdf(-abs(DM_stat), df=T - 1)

        # Construct named tuple for return
        dm_return = collections.namedtuple('dm_return', 'DM p_value')

        rt = dm_return(DM=DM_stat, p_value=p_value)

        return rt

    def NW_DM_test(self, pre1, pre2, y_test, datetime):
        '''DM检验
        输入：model1 prediction, model2 prediction, original return'''
        import statsmodels.formula.api as smf
        from scipy import stats
        import numpy as np
        from collections import namedtuple
        from scipy.stats import distributions

        def _ttest_finish(df, t):
            # from scipy.stats
            '''

            :param df:自由度
            :param t: t值
            :return: 输出t和对应p值
            '''
            prob = distributions.t.sf(np.abs(t), df) * 2  # use np.abs to get upper tail
            if t.ndim == 0:
                t = t[()]
            return t, prob

        def nwttest_1samp(a, popmean, axis=0, L=1):
            '''
            主函数
            :param a: 数据列表
            :param popmean: 原假设值u0
            :param axis: 行还是列，默认行
            :param L: lag， 滞后多少，默认1
            :return: 输出nw-t和对应p值
            '''
            a = np.array(a)
            N = len(a)
            df = N - 1
            e = a - np.mean(a)
            residuals = np.sum(e ** 2)
            Q = 0
            for i in range(L):
                w_l = 1 - (i + 1) / (1 + L)
                for j in range(1, N):
                    Q += w_l * e[j] * e[j - (i + 1)]
            S = residuals + 2 * Q
            nw_var = S / N
            d = np.mean(a, axis) - popmean
            nw_sd = np.sqrt(nw_var / float(df))
            with np.errstate(divide='ignore', invalid='ignore'):
                t = np.divide(d, nw_sd)
            t, prob = _ttest_finish(df, t)
            p = stats.t.sf(np.abs(t), len(a) - 1) * 2

            return nw_sd, p

        data = pd.DataFrame()
        pre1 = np.array(pre1).reshape(-1, )
        pre2 = np.array(pre2).reshape(-1, )
        data['pre1'] = np.array(pre1)
        data['pre2'] = np.array(pre2)
        data['y_test'] = np.array(y_test)
        data['date'] = np.array(datetime)
        data['date'] = pd.to_datetime(data['date'])
        d = data.groupby('date').apply(
            lambda x: np.mean((x['pre1'] - x['y_test']) ** 2 - (x['pre2'] - x['y_test']) ** 2))
        std, p = nwttest_1samp(d, 0)
        final_dm = np.mean(d) / std
        return final_dm, p

    def single_factor_analysis(self, data, size, start_year, end_year, end_month,
                               is_value=True):
        '''data-->DataFrame, 应该包括：factor,y_test,date
           end_month-->int, 最后一个时间点没有y_test，所以输入end_month用来判断'''
        data.columns = ['factor', 'y_test', 'date']
        data['date'] = pd.to_datetime(data['date'])
        group_ret = []
        if is_value:
            data['size'] = np.array(size)
            for year in range(start_year, end_year+1):
                for month in range(1, 13):
                    if year == end_year and month == end_month:
                        pass
                    else:
                        a = data[data['date'] == '{}-{}'.format(year, month)]
                        a.sort_values('factor', inplace=True)
                        mini = []
                        for i in range(10):
                            part = 10
                            b = a.iloc[0 + int(len(a) / part) * i:int(len(a) / part) * (i + 1), :]
                            mini.append(np.sum(b['y_test'] * (b['size'] / np.sum(b['size']))))
                        group_ret.append(mini)
                        print(year, month)
        else:
            for year in range(start_year, end_year+1):
                for month in range(1, 13):
                    if year == end_year and month == end_month:
                        pass
                    else:
                        a = data[data['date'] == '{}-{}'.format(year, month)]
                        a.sort_values('factor', inplace=True)
                        mini = []
                        for i in range(10):
                            part = 10
                            b = a.iloc[0 + int(len(a) / part) * i:int(len(a) / part) * (i + 1), :]
                            mini.append(np.mean(b['y_test']))
                        group_ret.append(mini)
                        print(year, month)
        group_ret = pd.DataFrame(group_ret)
        group_ret.columns = list(np.arange(1, 11))
        group_ret.index = data['date'].drop_duplicates(keep='first')
        group_ret['10-1'] = group_ret[10] - group_ret[1]

        final = pd.DataFrame(group_ret.mean())
        final.columns = ['mean_ret']
        from scipy import stats
        t_test = []
        for i in range(11):
            t_test.append(stats.ttest_1samp(group_ret.iloc[:, i], 0))
        final['ttest'] = t_test
        final['SR'] = group_ret.mean() * 12 / (group_ret.std() * np.power(12, 0.5))
        final = final.T
        return group_ret, final

    def MaxDrawdown(self, return_list):
        '''最大回撤率
           return_list是收益率列表，并不是收盘价列表'''
        i = np.argmax((np.maximum.accumulate(return_list) - return_list))  # 结束位置
        if i == 0:
            return 0
        j = np.argmax(return_list[:i])  # 开始位置
        return return_list[j] - return_list[i]

    def evaluation(self, ret):
        ret_mean = np.mean(ret)
        ret_std = np.std(ret)
        SR = ret_mean * 12 / (ret_std * np.power(12, 0.5))

        def MaxDrawdown(return_list):
            '''最大回撤率
               return_list是收益率列表，并不是收盘价列表'''
            i = np.argmax(
                (np.maximum.accumulate(return_list) - return_list) / np.maximum.accumulate(return_list))  # 结束位置
            if i == 0:
                return 0
            j = np.argmax(return_list[:i])  # 开始位置
            return return_list[j] - return_list[i]

        Maxdd = MaxDrawdown(ret)

        return ret_mean, SR, Maxdd

    def performance_analysis(self, time_period, y_true, y_pre, size, is_value=True):
        '''
        :param time_period: DataFrame.Series
        :param y_true: list or array
        :param y_pre: list or array
        :param size: list or array
        :param is_value: bool

        :return: float, float, float, DataFrame
        '''
        df = pd.DataFrame()
        df['date'] = time_period
        df['y_true'] = y_true
        df['y_pre'] = y_pre
        df['size'] = size
        long_df = df.groupby('date').apply(lambda x: x.sort_values('y_pre').tail(int(len(x)/10))).reset_index(drop=True)
        short_df = df.groupby('date').apply(lambda x: x.sort_values('y_pre').head(int(len(x)/10))).reset_index(drop=True)

        if is_value:
            long_df['weight'] = long_df.groupby('date').apply(lambda x: x['size']/x['size'].sum()).values
            short_df['weight'] = short_df.groupby('date').apply(lambda x: x['size']/x['size'].sum()).values

            long_ret = long_df.groupby('date').apply(lambda x: (x['y_true']*x['weight']).sum())
            short_ret = short_df.groupby('date').apply(lambda x: (x['y_true'] * x['weight']).sum())
        else:
            long_ret = long_df.groupby('date').apply(lambda x: x['y_true'].mean())
            short_ret = short_df.groupby('date').apply(lambda x: x['y_true'].mean())

        LS_ret = long_ret - short_ret
        LS_ret.index = long_ret.index

        mean_ret = LS_ret.mean()
        std = LS_ret.std()
        Sharpe_Ratio = mean_ret * 12 / (std * np.sqrt(12))

        cumsum_ret = LS_ret.cumsum()
        Max_dd = self.MaxDrawdown(cumsum_ret)

        # 仅多头
        long_mean_ret = long_ret.mean()
        long_std = long_ret.std()
        long_Sharpe_Ratio = long_mean_ret * 12 / (long_std * np.sqrt(12))

        long_cumsum_ret = long_ret.cumsum()
        long_Max_dd = self.MaxDrawdown(long_cumsum_ret)

        return mean_ret, std, Sharpe_Ratio, Max_dd, cumsum_ret, \
               long_mean_ret, long_std, long_Sharpe_Ratio, long_Max_dd, long_cumsum_ret