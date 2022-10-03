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
import time
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold,LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
import numpy as np
import matplotlib.pyplot as plt
start = time.process_time()


########################## 读入数据并分为trainSet和testSet ############################
def print_line(char) -> object:
    print(char*50)


########################## 读入数据并分为trainSet和testSet #################################
#data = pd.read_excel("path",sheet_name=0,index_col=None,header = [1],usecols=None)
import itertools as it
data=pd.read_excel(r'F:\\KNN\\数据\\压电性0919\\d33_0925.xls',sheet_name=2)
# data= np.array(data)
# # 将数据分成X和y
X = data.iloc[:,2:16]
print(X)
y = data.iloc[:,1]

print(X)

X_ti = X.columns.to_list()  #读取X的表头
print(X_ti)
for i in range(1,14,1):
    print(i)
    for e in it.combinations(X_ti, i):  # 输出的e为元组  # i为组合几个数
        X_c = list(e)  # 元组转化为list
        #print(X_c)
        # print(X[X_c])  #X[]  X[['小红','小李']] #选择两列，此时必须用[]将两列括起来
        X_train, X_test, y_train, y_test = train_test_split(X[X_c], y, test_size=0.3, random_state=0)
        # 将数据缩放至[0, 1]间。

        #############################调n_estimators##########################
        # ScoreAll = []
        # for i in range(2,20,1):
        #     regr_ET = ExtraTreesRegressor(n_estimators=183,
        #                                                   max_features=7,
        #                                     max_depth=15,
        #                                               min_samples_split=i,
        #                                      random_state=0)
        # # #     score = cross_val_score(regr_RF,X_train,y_train,cv=5).mean()
        # # #     ScoreAll.append([i,score])
        # # # ScoreAll = np.array(ScoreAll)
        # # #
        # #
        #
        #     regr_ET.fit(X_train,y_train)
        #     MSE = mean_squared_error(y_test,regr_ET.predict(X_test))
        #     ScoreAll.append([i,MSE])
        # ScoreAll = np.array(ScoreAll)
        # plt.figure(figsize=[20,5])
        # plt.plot(ScoreAll[:,0],ScoreAll[:,1])
        # plt.show()
        # min_MSE = np.where(ScoreAll==np.min(ScoreAll[:,1]))[0][0]
        # print("最优参数及最低误差：",ScoreAll[min_MSE])
        #############交叉验证###################
        kfold = KFold(n_splits=5, shuffle=True, random_state=0)
        loo = LeaveOneOut()
        regr_ET = ExtraTreesRegressor()
        scores = cross_val_score(regr_ET, X[X_c], y, cv=kfold, scoring='neg_root_mean_squared_error',n_jobs=-1)
        # print(scores)
        scores = scores.mean()
        print(X_c, scores,i)
    print_line("-")
        # ##########预测###############
        # regr_ET = ExtraTreesRegressor(min_samples_split=4,random_state=0)
        # regr_ET.fit(X_train, y_train)
        # y_test_pred_ET = regr_ET.predict(X_test)
        # y_train_pred_ET = regr_ET.predict(X_train)
        # print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_ET)))
        # print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_ET)))
        # score = r2_score(y_test, y_test_pred_ET)
        # print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_ET))
        # print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_ET))
        # print_line("-")

        # print('训练集数据')
        # for i, pred in enumerate(y_train_pred_ET):
        #     print("预测值：%s，实测值：%s" % (pred, y_train[i]))
        # #############测试集数据################
        # print('测试集数据')
        # for i, pred in enumerate(y_test_pred_ET):
        #     print("预测值：%s，实测值：%s" % (pred, y_test[i]))

        # unknown_data = pd.read_excel(r'F:\\KNN\\数据\\透明性0506\\课题组烧结温度data\\T.xlsx', sheet_name=5)
        # unknown_data = np.array(unknown_data)  # 将dataframe转化为array
        # x_pred = unknown_data[:, 4:14]
        # # std = Normalizer()
        # # x_pred = DataFrame(std.fit_transform(x_pred))
        # y_pred = regr_ET.predict(x_pred)
        # # y_pred =pd.DataFrame(y_pred)
        # print(list(y_pred))
        ##########################----绘图----###################################
        # plt.rcParams['font.sans-serif']=['simhei']   #用于正常显示中文标签
        # plt.rcParams['axes.unicode_minus']=False     #用于正常显示负号
        # fig = plt.figure(figsize=[5,5],dpi=300)
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
        # ax4 = fig.add_subplot(1,1,1)
        # ax4.scatter(y_train,y_train_pred_ET,color = 'b',label='训练集',s=30,marker='D')
        # ax4.scatter(y_test,y_test_pred_ET,color = 'r',label='测试集',s=30,marker='*')
        # ax4.set_title('ExtraTreeRegressor',size=20)
        # ax4.set_xlabel('测量值',fontsize = 15)
        # ax4.set_ylabel('预测值',fontsize = 15)
        # ax4.set_xlim(0,100)
        # ax4.set_ylim(0,100)
        # plt.xticks(fontsize=10)
        # ax4.plot(ax4.get_xlim(),ax4.get_ylim(),ls="--",c=".3")
        # plt.legend(loc='upper left')
        # plt.text(80,10, 'R^2= %.3f' % r2_score(y_test, y_test_pred_ET) ,bbox=dict(facecolor = 'yellow',alpha=0.5),fontsize=10)
        # # plt.tight_layout()
        # plt.show()

        ########################迭代特征选择################
        # from sklearn.feature_selection import RFE
        # for i in range(2,8,1):
        #     select = RFE(ExtraTreesRegressor(n_estimators=232,min_samples_split=4,random_state=0),
        #                  n_features_to_select=i)
        #     select.fit(X_train, y_train)
        #     rank = select.ranking_
        #     print(rank)
        #     mask = select.get_support()
        #     # plt.matshow(mask.reshape(1, -1), cmap='gray_r')
        #     # plt.xlabel("sample index")
        #     # plt.show()
        #     X_train_rfe = select.transform(X_train)
        #     X_test_rfe = select.transform(X_test)
        #     regr_ET = ExtraTreesRegressor(n_estimators=232,min_samples_split=4,random_state=0)
        #     regr_ET.fit(X_train_rfe, y_train)
        #     score = ExtraTreesRegressor(n_estimators=232,min_samples_split=4,random_state=0).fit(X_train_rfe, y_train).score(
        #         X_test_rfe, y_test )
        #     MSE = mean_squared_error(y_test, regr_ET.predict(X_test_rfe))
        #     print(----i)
        #     print(sqrt(MSE))
        #     print("test score: {:.3f}".format(score))

end = time.process_time()
print("Runing time", end - start)


