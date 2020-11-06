# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 16:21:44 2020

@author: 11004076
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import seaborn as sns

data['Ticket_info'] = data['Ticket'].apply(lambda x : 
    x.replace(".","").replace("/","").strip().split(' ')[0] if not x.isdigit() else 'X')
#%matplotlib inline  

np.random.seed(2) #

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep')


def ProdList(info):
    resp = requests.get(url + str(info), headers=headers)
    html = HTML(html=resp.text)
    return(html.find('a.product-list-item'))
============================================================    
from sklearn.ensemble import RandomForestRegressor
 
### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age','Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].iloc[:,:].values
    unknown_age = age_df[age_df.Age.isnull()].iloc[:,:].values

    # y即目标年龄
    y = known_age[:, 0] #矩陣的第0欄

    # X即特征属性值
    X = known_age[:, 1:] #矩陣的第1~最後欄

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)
    
    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])
    
    # 用得到的预测结果填补原缺失数据
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges 
    
    return df, rfr



def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train) 
data_train = set_Cabin_type(data_train) 
data_train


===================================================================


dataAgeNull = data[data["Age"].isnull()]
dataAgeNotNull = data[data["Age"].notnull()]
remove_outlier = dataAgeNotNull[(np.abs(dataAgeNotNull["Fare"]-dataAgeNotNull["Fare"].mean())>(4*dataAgeNotNull["Fare"].std()))|
                      (np.abs(dataAgeNotNull["Family_Size"]-dataAgeNotNull["Family_Size"].mean())>(4*dataAgeNotNull["Family_Size"].std()))                     
                     ]

rfModel_age = RandomForestRegressor(n_estimators=2000,random_state=42)
ageColumns = ['
rfModel_age.fit(remove_outlier[ageColumns], remove_outlier["Age"])

ageNullValues = rfModel_age.predict(X= dataAgeNull[ageColumns])
dataAgeNull.loc[:,"Age"] = ageNullValues
data = dataAgeNull.append(dataAgeNotNull)
data.reset_index(inplace=True, drop=True)
#
======================================================================
    ≤ 100 10%
    100 ＜ n ≤ 500 5%
    500 ＜ n ≤ 1000 1%
    n ＞ 2000 
--------

df = df.copy().sample(600)

from statsmodels.formula.api import ols


lm = ols('price ~ C(neighborhood) + C(style)', data=df).fit() 

lm = ols('price ~ area + bedrooms + bathrooms', data=df).fit()
lm.summary()
#
anova
tensorflow
keras
pytorch
bigquery
===========
# 自定義方差膨脹因子的檢測公式
def vif(df, col_i):
    """
    df: 整份數據
    col_i：被檢測的列名
    """
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)
###### 參數說​​明
y = df['broadband'] #他當下還遲疑了一下  我
X = df.iloc[:, 1:-1] 
# 客户 id 没有用，故丢弃 cust_id, 
## 0 可以表示第一列和最后一列，所以 1:-1 自动丢弃了第一列的客户 id 与最后一列的因变量

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size=0.4, random_state=12345)
## 根据原理部分，可知随机森林是处理数据不平衡问题的利器



model = Sequential() # 定義模型
model.add(Dense(units=64, activation='relu', input_dim=100)) # 定義網絡結構
model.add(Dense(units=10, activation='softmax')) # 定義網絡結構
model.compile(loss='categorical_crossentropy', # 定義loss函數、優化方法、評估標準
              optimizer='sgd',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=32) # 訓練模型
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128) # 評估模型
classes = model.predict(x_test, batch_size=128) # 使用訓練好的數據進行預測

import sklearn.tree as tree

import sklearn.tree as tree

# 直接使用交叉网格搜索来优化决策树模型，边训练边优化
from sklearn.model_selection import GridSearchCV

param_grid = {'criterion': ['entropy', 'gini'],
             'max_depth': [2, 3, 4, 5, 6, 7, 8],
             'min_samples_split': [4, 8, 12, 16, 20, 24, 28]} 
                # 通常来说，十几层的树已经是比较深了
clf = tree.DecisionTreeClassifier()  # 定义一棵树
clfcv = GridSearchCV(estimator=clf, param_grid=param_grid, scoring='roc_auc',
                    cv=4) # 传入模型，网格搜索的参数，评估指标，cv交叉验证的次数
                          ## 这里也只是定义，还没有开始训练模型

# I lift my voice to sing You praise
# No matter what life may bring my way
# I know that You are, the God of my life
# You are the One Who holds it all
# 
# I lift my hands to worship You
# Nowhere to find a love so true
# I know that You are, the God of my life
# Jesus I am crying out
# 
# Come Holy Spirit
# More of Your presence
# Take me as I am, use me as You call
# I give my all for You alone
# 
# Pour out Your Spirit
# Move in Your power
# Jesus You are all, all my soul longs for
# My heart is calling out to You
# (Jesus, my Lord)
# 
# Change and mould me
# To be more like You
# All my life I'll live to worship You
# Purify me
# Set my heart on fire
# Let me come through
# Pure as gold
# =============================================================================
