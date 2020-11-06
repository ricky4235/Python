# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:23:29 2020

@author: 11004076
"""



#三、進行數據處理
#1、缺失值

#（1）如果缺失的值很多佔樣本總數比例很高，則直接捨棄  

#讀取訓練集和測試集數據
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#先將訓練集和測試集合並，便於處理
tables = [train,test]
print(tables)
print("Delete features with high number of missing values...")
#統計每一個特徵的缺失數量，df.isnull()最終返回每個特徵下數值的缺失情況
total_missing = train.isnull().sum()
#to_delete存儲缺失量超過三分之一的屬性
to_delete = total_missing[total_missing > (train.shape[0]/3.)]
print(to_delete)

for table in tables:
    #刪除缺失值 超過總數據量三分之一的 屬性,使用drop函數刪除該列
    table.drop(list(to_delete.index),axis=1,inplace=True)
#（2）缺失的樣本個數適中，且該屬性非連續值特徵屬性，則將NaN作爲一個新類別，加到類別特徵中

#（3）缺失的樣本個數適中，且屬性值爲連續性特徵值，給定特定範圍，把它離散化，將NaN作爲一個type加到屬性類目中

（4）缺失的個數不多時，可以根據已有值，擬合數據補充 

def set_missing_ages(df):
    '''
    函數說明：使用RandomForestClassifier填充缺失的年齡的屬性
    函數實現：先將所有數值型的屬性提取出來，，根據目標屬性值劃分爲兩部分數據集進行擬合，再用來預測結果
    Parameters:
        df - DataFrame對象
    returns:
        data_train - 補全後的數據集
        rfr - 隨機森林擬合後的模型
    '''
    # 提取所有數值類型的屬性 數據
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
    print(age_df)
    # 將乘客分爲已知年齡和未知年齡兩部分
    known_age = age_df[age_df.Age.notnull()].values  # dataFrame.屬性.notnull()將Age屬性值值不爲空的數據提取出來
    unknown_age = age_df[age_df.Age.isnull()].values

    # y即目標年齡。所有行的第一列，對應的是年齡值
    y = known_age[:, 0]
    # x即特徵的屬性值，除了年齡外所有的特徵值提取出來
    x = known_age[:, 1:]

    # 將數據擬合到RandomForestRegressor中
    # 隨機森林的主要參數：max_features:允許單個決策樹使用特徵的最大數量；
    # n_estimators:在利用最大投票數或平均值來預測前，想要建立的子樹數量，較多的子樹性能更好，但是速度會變慢
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    # 使用隨機森林擬合數據，返回模型
    rfr.fit(x, y)
    # 用得到的模型進行未知年齡結果的預測
    predictedAges = rfr.predict(unknown_age[:, 1::])  # 取下標爲1的值，即取所有樣本的未知年齡值
    # 用得到的預測結果取填充之前缺失值
    df.loc[(data_train.Age.isnull()), 'Age'] = predictedAges  # 取Age屬性不爲空的所有值，補齊Age屬性值
    # 返回補全後的數據表，以及擬合的模型
    return data_train, rfr
2、分別篩選數值型和類目型的特徵，並將類目型特徵轉換爲one-hot編碼

#篩選數值類型的特徵
numerical_features = tran.select_dtypes(include=["float","int","bool"]).columns.values
print(numerical_features)

#篩選類目型的特徵
categorical_features = train.select_dtypes(include=["object"]).columns.values
print(categorical_features)
#將類目型的特徵使用get_dummies()轉換成one-hot編碼
dummies_Cabin = pd.get_dummies(data_train['Cabin'],prefix='Cabin')
#將編碼後的特徵列橫向拼接到原數據集上
df = pd.concat([data_train,dummies_Cabin],axis=1)
#刪除掉原來的列
df.drop(['Cabin'],axis=1,inplace=True)
return df
 

3、對於特徵值較大的特徵，爲了保證同等影響最終的分類結果，需要進行標準化

    #將Age和Fare的值標準化歸約到[-1,1]之間
    #定義標準化的對象
    scaler = preprocessing.StandardScaler()
    #擬合和轉換訓練的數據
    df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1,1))
    #擬合和轉換訓練的數據
    df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1,1))
4、刪除異常數據點

通過畫散點圖或者查看數據data.describe()，檢查數據中的異常點並刪除

train = train.drop(train[(train['ArLivArea']>4000)].index) #刪除ArLivArea>4000的樣本數據
四、Series類型和DataFrame類型說明
#Series相當於數組numpy.array，如果不爲其指定索引值，默認爲0，1，2...
s1 = pd.Series([1,2,3,4])
print(s1)
s2 = pd.Series([3,2,4,5],index=['a','b','c','d'])
print(s2)


print(s2.values)#返回Series序列值
print(s2.index)


#DataFrame是一個表格型的數據結構，DataFrame將Series使用的場景從一維拓展到多維，
# 既有行索引也有列索引
#1、創建dataFrame的方式：（列表、字典、Series、Numpy ndarrays、DataFrame類型）
#2、dataFrame的參數：dataframe[data,index,columns,dtypes,copy]
#data:各種形式的數據；index:行標籤；columns:指定列標籤，都可以使用列表；dtype:指定數值型數據的類型；
data = [['Alex',10],['Bob',12],['Claeke',13]]
df = pd.DataFrame(data,columns=['name','age'],dtype=float)
print(df)



#使用數組ndarrays來創建dataframe
data = {'Name':['Tom','Jack','lili'],'age':[28,32,15]}
df = pd.DataFrame(data,index=['1','2','3'])#指定索引值
print(df)



#使用字典創建dataframe,字典的鍵爲dataframe的列名，如果有的值爲空，則顯示NaN
data = [{'a':1,'b':2},{'a':5,'b':10,'c':20}]
df = pd.DataFrame(data,index=['first','second'])#定義行索引,a的first值是1，second值是5
print(df)
#也可以同時使用行索引和列索引創建
df = pd.DataFrame(data,index=['first','second'],columns=['a','b','c'])
print(df)



#使用序列值創建
data = {
    "one":pd.Series(["1","2","3"],index=["a","b","c"],dtype=float)
    "two":pd.Series(["1","2","3","4"],index=["a","b","c","d"])
}
df = pd.DataFrame(data)
print(df)



#使用numpy創建
pd.DataFrame(np.random.randint(60,100,size=(3,4)),index=["a","b","c"])


#DataFrame屬性
'''df.values:取出所有值
   df.index:行索引
   df.columns:列索引
   df.shape:表格的維度
#dataframe的切片
res.iloc[1,1:3]#取第二行，b-c列的數據
res.loc["A":"C","B":"C"]#取A-C行，B-C列的數據


#組合兩個dataframe對象，合併列
1、join()方法
data1.join(data2)
2、使用assign()方法添加一列
data1.assign('列名稱'=range(5))

#添加行
1、行合併兩個對象
pd.concat([data1,data2],ignore_index=True,sort=False)
2、使用append方法
data1.append(data2,sort=False)


#列刪除
1、刪除列,無返回值，直接改變原對象
del(df['one'])
2、pop方法，只刪除列，有返回值，不修改原對象


#drop()方法最常用，默認刪除行，可以傳入行索引名刪除指定的行
#刪除列的方法相同，但是需要設置axis=1


df.iloc[1:2,3:4]#使用索引值進行定位，取1-2行，3-4列數據
df.loc['name1':'name3','age1':'age3']#可以使用index和索引名進行定位