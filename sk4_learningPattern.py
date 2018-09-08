import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
'''
引入鸢尾花数据集
'''
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

#[:2,:] print 2个sample，所有的属性
# print(iris_X[:2,:])
# #共有三个类别
# print(iris_y)

# train_test_split(train_data,train_target,test_size=)
# 返回四组数据X_train,x_test,y_train,y_test
X_train,X_test,y_train,y_test = train_test_split(iris_X,iris_y,test_size=0.3)

# print(y_train)

'''
使用SKLEARN中的KNN模型
'''
knn = KNeighborsClassifier()
# fit中喂入训练数据，自动完成train的过程
knn.fit(X_train,y_train)

prediction = knn.predict(X_test)

print(prediction)
print(y_test)

accuracy = np.sum(1-np.equal(prediction,y_test).astype(float))/y_test.shape[0]

print (accuracy)