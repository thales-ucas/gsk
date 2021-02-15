import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.dates as mdates

from statsmodels.tsa.arima_model import ARMA, ARIMA
from statsmodels.graphics.api import qqplot
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import math
import os
import copy
import itertools
import seaborn as sns #热力图

plt.rcParams['font.family'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class TimeSeries:
    def __init__(self, df, name):
        """
        时间序列分析
        """
        self.origin = df
        self.name = name
        self.data = self.getSeries()
        self.dataset = [self.data]
        self.armaModel = None
        self.step = 0
    def getSeries(self):
        """
        按照时间序列需求处理数据
        """
        data = pd.DataFrame({'timeSeries': self.origin['营业总收入']})
        data.index = self.origin['时间']
        data['timeSeries'] = data['timeSeries'].map(lambda x: x / 1000000)
        data.columns = [self.name]
        return data
    def baseShow(self, step=0):
        """
        基本显示
        """
        df = self.dataset[step]
        fig, ax = plt.subplots(figsize=(15, 6))
        # ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.plot(df)
        ax.set_ylabel('营业总收入(百万英镑)')
        fig.suptitle(self.name)
    def acfShow(self, step=0):
        """
        自相关性
        """
        df = self.dataset[step]
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        plot_acf(df, ax=axes[0], lags=4)
        plot_pacf(df, ax=axes[1], lags=4)
        fig.suptitle(self.name)
        plt.show()
    def dickey(self, step=0):
        """
        单位跟检验
        """
        t = adfuller(self.dataset[step])
        output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
        output['value']['Test Statistic Value'] = t[0]
        output['value']['p-value'] = t[1]
        output['value']['Lags Used'] = t[2]
        output['value']['Number of Observations Used'] = t[3]
        output['value']['Critical Value(1%)'] = t[4]['1%']
        output['value']['Critical Value(5%)'] = t[4]['5%']
        output['value']['Critical Value(10%)'] = t[4]['10%']
        return output
    def thermodynamicOrder(self, ar=4, ma=2, step=0):
        """
        order热力图
        """
        # 创建Dataframe,以BIC准则
        results_aic = pd.DataFrame(\
            index=['AR{}'.format(i) for i in range(0, ar+1)],\
            columns=['MA{}'.format(i) for i in range(0, ma+1)])
        for p, q in itertools.product(range(0, ar+1), range(0, ma+1)):
            if p==0 and q==0:
                results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
                continue
            try:
                results = self.arma((p, q), step)
                #返回不同pq下的model的BIC值
                results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.aic
            except:
                continue
        results_aic = results_aic[results_aic.columns].astype(float)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax = sns.heatmap(results_aic,
                        #mask=results_aic.isnull(),
                        ax=ax,
                        annot=True, #将数字显示在热力图上
                        fmt='.2f',
                        )
        ax.set_title('AIC')
        plt.show()
    def arma(self, order, step=0):
        """
        创建arma模型
        """
        self.step = step
        self.armaModel = ARMA(self.dataset[step], order).fit()
        return self.armaModel
    def qqShow(self, step=0):
        """
        qq图
        """
        resid = self.armaModel.resid
        fig, ax = plt.subplots(figsize=(10, 6))
        qqplot(resid,line='q',ax=ax,fit=True)
        plt.show()
    def predict(self, start, end, step=0):
        """
        预测
        """
        series = self.data[self.name]
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.plot(self.data, 'r', label='原始数据')
        if step > 0:
            ax.plot(self.armaModel.fittedvalues.index, self.revert(self.armaModel.fittedvalues, self.data[self.name][0]), 'g', label='ARMA模型')
        else:
            ax.plot(self.armaModel.fittedvalues, 'g', label='ARMA模型')
        future = self.armaModel.predict(start=start, end=end)
        if step > 0:
            ax.plot(future.index, self.revert(future, self.data[self.name][-1]), 'b', label='预测数据')
        else:
            ax.plot(future, 'b', label='预测数据')
        ax.legend()
        ax.set_ylabel('营业总收入(百万英镑)')
        fig.suptitle(self.name)
    def revert(self, diffValues, *lastValue):
        for i in range(len(lastValue)):
            result = []
            lv = lastValue[i]
            for dv in diffValues:
                lv = dv + lv
                result.append(lv)
            diffValues = result
        return diffValues
    def diff(self):
        """
        差分
        """
        df = self.dataset[len(self.dataset)- 1]
        D_data = df.diff().dropna()
        self.dataset.append(D_data)