from sklearn import svm
from sklearn import datasets
'''
保存已经训练好的模型
'''
clf = svm.SVC()
iris = datasets.load_iris()
X,y=iris.data,iris.target
clf.fit(X,y)

'''
方法一：使用pickle保存数据
'''
# 保存模型
# import pickle
# with open('save/clf.pickle','wb') as f:
#     pickle.dump(clf,f)
#
# # 导出模型
# with open('save/clf.pickle','rb') as f:
#     clf2= pickle.load(f)
#     print(clf2.predict(X[0:1]))

'''
方法二：使用sklean的joblib保存数据
'''
from sklearn.externals import joblib

#保存模型
joblib.dump(clf,'save/clf.pkl')
#导出模型
clf3 = joblib.load('save/clf.pkl')
print(clf3.predict(X[0:1]))