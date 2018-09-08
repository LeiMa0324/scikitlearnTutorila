from  sklearn.model_selection import learning_curve
from sklearn.datasets import load_digits
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

'''
函数目的：查看随着样本增多，train_loss和test_loss的表现
如果train一直优于test，则有过拟合的风险
train_loss 的shape为[train_sizes.shape,cv]，使用十折交叉验证，在每一份train_size下得到train_loss和test_loss
将每一份size下的十折交叉验证的准确率求平均后比较变化曲线
'''

train_size,train_loss,test_loss = learning_curve(SVC(gamma=0.01),
                                                 X,
                                                 y,
                                                 cv=10,
                                                 scoring='mean_squared_error',
                                                 train_sizes=[0.1,0.25,0.5,0.75,1])

train_loss_mean = -np.mean(train_loss,axis=1)
test_loss_mean = -np.mean(test_loss,axis=1)

plt.plot(train_size,train_loss_mean,'o-',color='r',label="Training")
plt.plot(train_size,test_loss_mean,'o-',color='g',label="Cross-validation")

plt.xlabel("Training examples")
plt.ylabel("Loss")
plt.legend(loc="best")
plt.show()