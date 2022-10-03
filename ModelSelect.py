from pandas import DataFrame
from math import sqrt
import pandas as pd
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C,RBF,WhiteKernel as W
from sklearn.cross_decomposition import PLSRegression

from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold,LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
import numpy as np
import matplotlib.pyplot as plt

def print_line(char) -> object:
    print(char*50)

########################## 读入数据并分为trainSet和testSet #################################
#data = pd.read_excel("path",sheet_name=0,index_col=None,header = [1],usecols=None)
data=pd.read_excel(r'F:\\KNN\\数据\\压电性0919\\SISSOstar.xls',sheet_name=0)
data = np.array(data)#将dataframe转化为array

# 将数据分成X和y
#X = data.iloc[:,1:-1]
#y = data.iloc[:,-1]
X = data[:,2:36]  #开始要减一、结束不用减
y = data[:,1]

print(X)
print(y)

# 将数据缩放至[0, 1]间。
# std = StandardScaler()
# X = DataFrame(std.fit_transform(x))

################################### 选择模型 ############################################
regr_LR = linear_model.LinearRegression()
regr_RCV = linear_model.RidgeCV(alphas=[0.01, 0.05, 0.1, 1.0, 5.0, 10.0])
regr_BR = linear_model.BayesianRidge()
regr_KR = KernelRidge(kernel='rbf')
regr_RF = RandomForestRegressor()
regr_SVR = SVR()
# 新加入model
regr_R = Ridge(alpha=0.01)
regr_GBR = GradientBoostingRegressor(learning_rate=0.5)
regr_ETR = ExtraTreesRegressor()
# 核函数取值
kernel = W(noise_level=0.6) + 0.00316 ** 2 * RBF(length_scale=109)  # 0.72/0.56
regr_GPR = GaussianProcessRegressor(kernel=kernel,
                                    n_restarts_optimizer=10, alpha=0.1)
regr_PLS = PLSRegression(n_components=8)
# LinearRegression

regr_LR.fit(X_train, y_train)
y_test_pred_LR = regr_LR.predict(X_test)
y_train_pred_LR = regr_LR.predict(X_train)
print('LinearRegression:')
scores = cross_val_score(regr_LR,X,y,cv=10)
print('Cross_validation scores: {}'.format(scores.mean()))
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_LR)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_LR)))
print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_LR))
print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_LR))
print_line("-")

# RidgeCV
regr_RCV.fit(X_train, y_train)
y_test_pred_RCV = regr_RCV.predict(X_test)
y_train_pred_RCV = regr_RCV.predict(X_train)
print('RidgeCV:')
scores = cross_val_score(regr_RCV,X,y,cv=20)
print('Cross_validation scores: {}'.format(scores.mean()))
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_RCV)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_RCV)))
print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_RCV))
print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_RCV))
print_line("-")

# BayesianRidge
regr_BR.fit(X_train, y_train)
y_test_pred_BR = regr_BR.predict(X_test)
y_train_pred_BR = regr_BR.predict(X_train)
print('BayesianRidge:')
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_BR)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_BR)))
print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_BR))
print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_BR))
print_line("-")

# KernelRidge
regr_KR.fit(X_train, y_train)
y_test_pred_KR = regr_KR.predict(X_test)
y_train_pred_KR = regr_KR.predict(X_train)
print('KernelRidge:')
scores = cross_val_score(regr_KR,X,y,cv=10)
print('Cross_validation scores: {}'.format(scores.mean()))
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_KR)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_KR)))
print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_KR))
print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_KR))
print_line("-")

# RandomForestRegressor
regr_RF.fit(X_train, y_train)
y_test_pred_RF = regr_RF.predict(X_test)
y_train_pred_RF = regr_RF.predict(X_train)
loo = LeaveOneOut()
scores = cross_val_score(regr_RF,X,y,cv=5)
print('Cross_validation scores: {}'.format(scores.mean()))
print('RandomForestRegressor:')
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_RF)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_RF)))
print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_RF))
print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_RF))

print_line("-")

# SVR
regr_SVR.fit(X_train, y_train)
y_test_pred_SVR = regr_SVR.predict(X_test)
y_train_pred_SVR = regr_SVR.predict(X_train)
print('SVR:')
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_SVR)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_SVR)))
score = r2_score(y_test, y_test_pred_SVR)
print(score)
scores = cross_val_score(regr_SVR,X,y,cv=10)
print('Cross_validation scores: {}'.format(scores.mean()))
print('Training Variance score: %.5f' % r2_score(y_train, y_train_pred_SVR))
print('Testing Variance score: %.5f' % r2_score(y_test, y_test_pred_SVR))
print_line("-")
# print('Variance score: %.2f' % regr_SVR.score(X_test, y_test))  #作用同上

# Ridge
regr_R.fit(X_train, y_train)
y_test_pred_R = regr_R.predict(X_test)
y_train_pred_R = regr_R.predict(X_train)
print('Ridge:')
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_R)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_R)))
score = r2_score(y_test, y_test_pred_R)
print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_R))
print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_R))
print_line("-")

# GradientBoostingRegressor
regr_GBR.fit(X_train, y_train)
y_test_pred_GBR = regr_GBR.predict(X_test)
y_train_pred_GBR = regr_GBR.predict(X_train)
print('GBR:')
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_GBR)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_GBR)))
score = r2_score(y_test, y_test_pred_GBR)
scores = cross_val_score(regr_GBR,X,y,cv=10)
print('Cross_validation scores: {}'.format(scores.mean()))
print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_GBR))
print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_GBR))
print_line("-")

# ExtraTreesRegressor
regr_ETR.fit(X_train, y_train)
y_test_pred_ETR = regr_ETR.predict(X_test)
y_train_pred_ETR = regr_ETR.predict(X_train)
print('ETR:')
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_ETR)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_ETR)))
print("train_MAE: %.2f" % mean_absolute_error(y_train, y_train_pred_ETR))
print("test_MAE: %.2f" % mean_absolute_error(y_test, y_test_pred_ETR))
score = r2_score(y_test, y_test_pred_ETR)
print(regr_ETR.n_estimators, regr_ETR.max_depth, regr_ETR.max_features, regr_ETR.min_samples_split,
      regr_ETR.max_leaf_nodes)
print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_ETR))
print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_ETR))
print_line("-")

# GaussianProcessRegressor
regr_GPR.fit(X_train, y_train)
y_test_pred_GPR = regr_GPR.predict(X_test)
y_train_pred_GPR = regr_GPR.predict(X_train)
print('GPR:')
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_GPR)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_GPR)))
score = r2_score(y_test, y_test_pred_GPR)
print(score)
print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_GPR))
print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_GPR))
print_line("-")

# PLSRegression
regr_PLS.fit(X_train, y_train)
y_test_pred_PLS = regr_PLS.predict(X_test)
y_train_pred_PLS = regr_PLS.predict(X_train)
print('PLS:')
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_PLS)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_PLS)))
score = r2_score(y_test, y_test_pred_PLS)
print(score)
print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_PLS))
print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_PLS))
print_line("-")



# ##########################绘图###################################
plt.rcParams['font.sans-serif']=['simhei']   #用于正常显示中文标签
plt.rcParams['axes.unicode_minus']=False     #用于正常显示负号
fig = plt.figure(figsize=[20,20],dpi=100)
#
#
# # 第一个参数是x轴坐标
# # 第二个参数是y轴坐标
# # 第三个参数是要显式的内容
# # alpha 设置字体的透明度
# # family 设置字体
# # size 设置字体的大小
# # style 设置字体的风格
# # wight 字体的粗细
# # bbox 给字体添加框，alpha 设置框体的透明度， facecolor 设置框体的颜色
#
ax1 = fig.add_subplot(3,3,1)
# ax1.scatter(y_train,y_train_pred_RCV,color = 'b',label='训练集')
# ax1.scatter(y_test,y_test_pred_RCV,color = 'r',label='测试集')
# ax1.set_title('RidgeCV',size=20)
# ax1.set_xlabel('测量值',fontsize = 15)
# ax1.set_ylabel('预测值',fontsize = 15)
# ax1.set_xlim(0,1)
# ax1.set_ylim(0,1)
# ax1.plot(ax1.get_xlim(),ax1.get_ylim(),ls="--",c=".3")
# plt.legend(loc='upper left')
# plt.text(450,150, 'R^2= %.3f' % r2_score(y_test, y_test_pred_RCV) ,bbox=dict(facecolor = 'yellow',alpha=0.5),fontsize=20)

# ax2 = fig.add_subplot(3,3,2)
# ax2.scatter(y_train,y_train_pred_BR,color = 'b',label='训练集')
# ax2.scatter(y_test,y_test_pred_BR,color = 'r',label='测试集')
# ax2.set_title('BayesianRidge',size=20)
# ax2.set_xlabel('测量值',fontsize = 15)
# ax2.set_ylabel('预测值',fontsize = 15)
# ax2.set_xlim(100,600)
# ax2.set_ylim(100,600)
# ax2.plot(ax1.get_xlim(),ax1.get_ylim(),ls="--",c=".3")
# plt.legend(loc='upper left')
# plt.text(450,150, 'R^2= %.3f' % r2_score(y_test, y_test_pred_BR) ,bbox=dict(facecolor = 'yellow',alpha=0.5),fontsize=20)
#
# ax3 = fig.add_subplot(3,3,3)
# ax3.scatter(y_train,y_train_pred_KR,color = 'b',label='训练集')
# ax3.scatter(y_test,y_test_pred_KR,color = 'r',label='测试集')
# ax3.set_title('KernelRidge',size=20)
# ax3.set_xlabel('测量值',fontsize = 15)
# ax3.set_ylabel('预测值',fontsize = 15)
# ax3.set_xlim(100,600)
# ax3.set_ylim(100,600)
# ax3.plot(ax1.get_xlim(),ax1.get_ylim(),ls="--",c=".3")
# plt.legend(loc='upper left')
# plt.text(450,150, 'R^2= %.3f' % r2_score(y_test, y_test_pred_KR) ,bbox=dict(facecolor = 'yellow',alpha=0.5),fontsize=20)
#
ax4 = fig.add_subplot(3,3,2)
ax4.scatter(y_train,y_train_pred_RF,color = 'b',label='训练集')
ax4.scatter(y_test,y_test_pred_RF,color = 'r',label='测试集')
ax4.set_title('RandomForestRegressor',size=20)
ax4.set_xlabel('测量值',fontsize = 15)
ax4.set_ylabel('预测值',fontsize = 15)
ax4.set_xlim(0,1)
ax4.set_ylim(0,1)
ax4.plot(ax1.get_xlim(),ax1.get_ylim(),ls="--",c=".3")
plt.legend(loc='upper left')
plt.text(450,150, 'R^2= %.3f' % r2_score(y_test, y_test_pred_RF) ,bbox=dict(facecolor = 'yellow',alpha=0.5),fontsize=20)

# ax6 = fig.add_subplot(3,3,5)
# ax6.scatter(y_train,y_train_pred_SVR,color = 'b',label='训练集')
# ax6.scatter(y_test,y_test_pred_SVR,color = 'r',label='测试集')
# ax6.set_title('SVR',size=20)
# ax6.set_xlabel('测量值',fontsize = 10)
# ax6.set_ylabel('预测值',fontsize = 10)
# ax6.set_xlim(100,600)
# ax6.set_ylim(100,600)
# ax6.plot(ax1.get_xlim(),ax1.get_ylim(),ls="--",c=".3")
# plt.legend(loc='upper left')
# plt.text(450,150, 'R^2= %.3f' % r2_score(y_test, y_test_pred_SVR) ,bbox=dict(facecolor = 'yellow',alpha=0.5),fontsize=20)
#
# ax7 = fig.add_subplot(3,3,6)
# ax7.scatter(y_train,y_train_pred_R,color = 'b',label='训练集')
# ax7.scatter(y_test,y_test_pred_R,color = 'r',label='测试集')
# ax7.set_title('Ridge',size=20)
# ax7.set_xlabel('测量值',fontsize = 10)
# ax7.set_ylabel('预测值',fontsize = 10)
# ax7.set_xlim(100,600)
# ax7.set_ylim(100,600)
# ax7.plot(ax1.get_xlim(),ax1.get_ylim(),ls="--",c=".3")
# plt.legend(loc='upper left')
# plt.text(450,150, 'R^2= %.3f' % r2_score(y_test, y_test_pred_R) ,bbox=dict(facecolor = 'yellow',alpha=0.5),fontsize=20)
#
ax8 = fig.add_subplot(3,3,3)
ax8.scatter(y_train,y_train_pred_GBR,color = 'b',label='训练集')
ax8.scatter(y_test,y_test_pred_GBR,color = 'r',label='测试集')
ax8.set_title('GradientBoostingRegressor',size=20)
ax8.set_xlabel('测量值',fontsize = 10)
ax8.set_ylabel('预测值',fontsize = 10)
ax8.set_xlim(0,1)
ax8.set_ylim(0,1)
ax8.plot(ax1.get_xlim(),ax1.get_ylim(),ls="--",c=".3")
plt.legend(loc='upper left')
plt.text(450,150, 'R^2= %.3f' % r2_score(y_test, y_test_pred_GBR) ,bbox=dict(facecolor = 'yellow',alpha=0.5),fontsize=20)

ax9 = fig.add_subplot(3,3,1)
ax9.scatter(y_train,y_train_pred_ETR,color = 'b',label='训练集')
ax9.scatter(y_test,y_test_pred_ETR,color = 'r',label='测试集')
ax9.set_title('ExtraTreeRegressor',size=20)
ax9.set_xlabel('测量值',fontsize = 10)
ax9.set_ylabel('预测值',fontsize = 10)
ax9.set_xlim(0,1)
ax9.set_ylim(0,1)
ax9.plot(ax1.get_xlim(),ax1.get_ylim(),ls="--",c=".3")
plt.legend(loc='upper left')
plt.text(450,150, 'R^2= %.3f' % r2_score(y_test, y_test_pred_ETR) ,bbox=dict(facecolor = 'yellow',alpha=0.5),fontsize=20)

# ax10 = fig.add_subplot(3,3,9)
# ax10.scatter(y_train,y_train_pred_GPR,color = 'b',label='训练集')
# ax10.scatter(y_test,y_test_pred_GPR,color = 'r',label='测试集')
# ax10.set_title('Gaussian Rocess Regressor',size=20)
# ax10.set_xlabel('测量值',fontsize = 10)
# ax10.set_ylabel('预测值',fontsize = 10)
# ax10.set_xlim(100,600)
# ax10.set_ylim(100,600)
# ax10.plot(ax1.get_xlim(),ax1.get_ylim(),ls="--",c=".3")
# plt.legend(loc='upper left')
# plt.text(450,150, 'R^2= %.3f' % r2_score(y_test, y_test_pred_GPR) ,bbox=dict(facecolor = 'yellow',alpha=0.5),fontsize=20)
#
plt.tight_layout()
plt.show()
