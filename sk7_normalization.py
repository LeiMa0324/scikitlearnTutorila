from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification
# support vector classifier
from sklearn.svm import  SVC
import matplotlib.pyplot as plt

# a = np.array([[10,2.7,3.6],
#               [-100,5,-2],
#               [120,20,40]],dtype=np.float64)
# print (a)
#
# # 零均值单位方差
# print(preprocessing.scale(a))

X,y = make_classification(n_samples=200,    #样本个数
                          n_features=2,     #特征个数，2个
                          n_redundant=0,    #冗余特征个数（有效特征线性组合）
                          n_informative=2,  #有效特征个数
                          n_repeated=0,     #重复特征个数（有效特征和冗余的线性组合）
                          random_state=22,  #random generator产生22个seed
                          n_clusters_per_class=1,   #簇的个数
                          scale=100)    #feature为[1,100]*scale
print (X,y)

# # 绘制第一个属性和第二个属性的散点图，用y区分颜色
# plt.scatter(X[:,0],X[:,1],c=y)
# plt.show()

'''
使用scale标准化X的各维度取值范围后，精确度为0.95
不标准化，精确度为0.5
'''
X = preprocessing.scale(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)
#使用SVM做分类
clf = SVC()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))