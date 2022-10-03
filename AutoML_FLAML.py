from pandas import DataFrame
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score,KFold,LeaveOneOut
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn.preprocessing import MinMaxScaler,StandardScaler,Normalizer
import numpy as np
from flaml import AutoML

########################## 读入数据并分为trainSet和testSet #################################
#data = pd.read_excel("path",sheet_name=0,index_col=None,header = [1],usecols=None)
data=pd.read_excel(r'F:\\KNN\\EV-d33-KNN paper\\papers\\d33_0925.xls',sheet_name=1)
X = data.iloc[:,2:31]
y = data.iloc[:,1]

print(X)
print(y)

# 将数据缩放至[0, 1]间。
# std = StandardScaler()
# X = DataFrame(std.fit_transform(x))

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

automl = AutoML()
settings = {
    "time_budget": 240,
    "metric" : 'r2',
    "task": 'regression',
    "log_file_name": 'airlines_experiment.log',
    "seed": 7654321
}
automl.fit(X_train=X_train, y_train=y_train, **settings)
print('Best ML leaner:', automl.best_estimator)
print('Best hyperparmeter config:', automl.best_config)
print('Best r2 on validation data: {0:.4g}'.format(1-automl.best_loss))
print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))
