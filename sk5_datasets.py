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
print(data_X)
print(data_X_std)
'''
选择一个模型
model.fit(X_train,y_train)即为学习
model.predict(X_test)即为预测
'''
model = LinearRegression()
model.fit(data_X_std,data_y)
'''
输出预测值
'''
#print data中前4个值
print(model.predict(data_X_std[:4,:]))
print(data_y[:4])

'''
例子2：创造数据
一维x和一维y的数据
'''

X,y = datasets.make_regression(n_samples=100,n_features=1,n_targets=1,noise=10)
'''
绘制散点图
'''
plt.scatter(X,y)
plt.show()

