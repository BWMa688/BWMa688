from pandas import DataFrame
from math import sqrt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor
import shap
import time
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold,LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer,RobustScaler
import numpy as np
import matplotlib.pyplot as plt
start = time.process_time()
########################## 读入数据并分为trainSet和testSet ############################
def print_line(char) -> object:
    print(char*50)

#####
#data = pd.read_excel("path",sheet_name=0,index_col=None,header = [1],usecols=None)
data=pd.read_excel(r'F:\\KNN\\数据\\压电性0919\\d33_0925.xls',sheet_name=1)
features = data.columns[2:35].tolist()#提取特征集
print(features)

data = np.array(data)#将dataframe转化为array
X = data[:,2:31]  #开始要减一、结束不用减
y = data[:,1]
print(X)
print(y)
# std = Normalizer()
# X = DataFrame(std.fit_transform(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


#############################调n_estimators##########################
ScoreAll = []
for i in range(2,30,1):
    regr_RF = RandomForestRegressor(n_estimators=i,
                                                 #max_features=i,
                                    #max_depth=i,
                                                # min_samples_split=i,
                                     random_state=0)
# #     score = cross_val_score(regr_RF,X_train,y_train,cv=5).mean()
# #     ScoreAll.append([i,score])
# # ScoreAll = np.array(ScoreAll)
    regr_RF.fit(X_train,y_train)
    RMSE = sqrt(mean_squared_error(y_test,regr_RF.predict(X_test)))
    ScoreAll.append([i,RMSE])
plt.figure(figsize=[20,5])
plt.plot(ScoreAll[:,0],ScoreAll[:,1])
plt.show()
#############交叉验证###################
kfold =KFold(n_splits=5,shuffle = True,random_state=0)
loo = LeaveOneOut()
# ###########预测###############

regr_ET = RandomForestRegressor(n_estimators=132,random_state=0)
model = regr_ET.fit(X_train, y_train)
y_test_pred_ET = regr_ET.predict(X_test)
y_train_pred_ET = regr_ET.predict(X_train)
print('ETR:')
print("train_RMSE: %.2f" % sqrt(mean_squared_error(y_train, y_train_pred_ET)))
print("test_RMSE: %.2f" % sqrt(mean_squared_error(y_test, y_test_pred_ET)))
print("train_MAE: %.2f" % mean_absolute_error(y_train, y_train_pred_ET))
print("test_MAE: %.2f" % mean_absolute_error(y_test, y_test_pred_ET))
score = r2_score(y_test, y_test_pred_ET)
print('Training Variance score: %.2f' % r2_score(y_train, y_train_pred_ET))
print('Testing Variance score: %.2f' % r2_score(y_test, y_test_pred_ET))
print_line("-")

############测试集数据################
print('测试集数据')
for i, pred in enumerate(y_test_pred_ET):
    print("预测值：%s，实测值：%s" % (pred, y_test[i]))


####SHAP某个特征大小与预测值之间的关系###
# explainer = shap.Explainer(model)
# shap_values = explainer(X_train)
# #print(shap_values)
# # f = plt.figure(dpi=600,figsize=(10,10))
# shap.summary_plot(shap_values,X_train,feature_names=features,alpha=0.8)   #bee图
# # #plt.savefig = ("PDFbee.png")
# shap.summary_plot(shap_values,X,feature_names=features, plot_type="bar")  ###柱状图


# shap.dependence_plot("ST", shap_values, X_train,interaction_index='EA')
# shap.plots.scatter(shap_values[:,0],alpha = 0.8,color=shap_values[:,2:3])#特征交互分析
# print(shap_values)
# df = pd.DataFrame(shap_values.data)
# print(df)
# df.to_excel("shap_values_data.xlsx")

#######new material######
# unknown_data = pd.read_excel(r'F:\\KNN\\数据\\透明性0506\\课题组烧结温度data\\T.xlsx',sheet_name=3)
# unknown_data = np.array(unknown_data)  # 将dataframe转化为array
# x_pred = unknown_data[:, 4:15]
# # std = Normalizer()
# # x_pred = DataFrame(std.fit_transform(x_pred))
# y_pred = regr_RF.predict(x_pred)
# print(y_pred)
# # ##########################----绘图----###################################
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
# ax4.set_xlim(0,1)
# ax4.set_ylim(0,1)
# plt.xticks(fontsize=10)
# ax4.plot(ax4.get_xlim(),ax4.get_ylim(),ls="--",c=".3")
# plt.legend(loc='upper left')
# plt.text(0.8,0.1, 'R^2= %.3f' % r2_score(y_test, y_test_pred_ET) ,bbox=dict(facecolor = 'yellow',alpha=0.5),fontsize=10)
# # plt.tight_layout()
# plt.show()

########################迭代特征选择################


################################### 拟合画图 ############################################




end = time.process_time()
print("Runing time", end - start)




