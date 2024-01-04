from gplearn.genetic import SymbolicRegressor
from pandas import DataFrame
from math import sqrt
from gplearn.genetic import SymbolicRegressor
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,ExtraTreesRegressor
import time
import shap
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
X = data[['ST', 'No6', 'No10', 'No41', 'No54\\3', 'No65', 'No67', 't'] ]
print(X)
y = data.iloc[:,1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=0)
param_grid = [
    {'p_crossover':[0.5,0.525,0.55,0.575,0.6,0.625,0.65,0.675,0.7,0.725,0.75,0.775,0.8,0.825,0.85,0.875,0.9,0.925,0.95],
     'p_subtree_mutation':[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09],
     'parsimony_coefficient':[0.0005,0.001,0.0015]}
]
est_gp = SymbolicRegressor(population_size=5000,
                           generations=20, stopping_criteria=60,
                           p_hoist_mutation='p_subtree_mutation', p_point_mutation=1,
                           max_samples=0.9, verbose=1, random_state=0)
grid_search = GridSearchCV(est_gp, param_grid, cv=5,scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(est_gp._program)
