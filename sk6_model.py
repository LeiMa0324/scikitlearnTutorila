import numpy as np
from sklearn import datasets
#引入线性回归模型
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

'''
例子1：引入波士顿房价数据集
'''
loaded_data = datasets.load_boston()
# 所有的数据集 .data为输入数据集，.target为标签
data_X = loaded_data.data
data_y = loaded_data.target

'''
归一化Normailization:将特征通过最大最小值，缩放到[0,1]之间
标准化standardization：将特征值缩放为一个标准正态分布，均值为0，方差为1
'''
std = StandardScaler()
data_X_std = std.fit_transform(data_X)

'''
选择一个模型
model.fit(X_train,y_train)即为学习
model.predict(X_test)即为预测
'''
model = LinearRegression()
model.fit(data_X,data_y)
'''
model的属性和功能
'''
#输出参数(weights)
print(model.coef_) #y = 0.1x+0.3
#输出截距(bias)
print(model.intercept_)
#返回调用LinearRegression时的参数值，不填则为默认
print(model.get_params())
# 返回R^2 COEFFICIENT OF DETERMINATION 决定系数
print(model.score(data_X,data_y))