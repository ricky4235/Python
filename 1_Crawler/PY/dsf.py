# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 15:23:29 2020

@author: 11004076
"""

在Kaggle比賽中最終要的步驟就是進行數據的分析，數據的清洗，以及特徵的提取。
因此我總結了最近常會用到的數據處理的方法，以便將來複習和使用。

一、讀取和存儲csv文件
從.csv文件中讀取文件內容；將DataFrame對象存放到.csv文件中

#讀取文件內容
train = pd.read_csv('train.csv',index_col=0)#讀取內容時，如果每行前面有索引值，捨去

#將DataFrame類型的對象（最終提交的結果形式），轉換爲.csv文件
result = pd.DataFrame({'PassengerId':data_test['PassengerId'].values,'Survived':prediction.astype(np.int32)})
result.to_csv("predict_result.csv",index=False) #每行的內容前不加索引
二、數據分析
1、對於各個特徵，可以通過畫條形圖來顯示每個特徵與分類標籤之間的關係，例如在Titanic比賽中，
觀察每個特徵與獲救情況之間的關係

#plt.subplot2grid((m,n),(a,b))：劃分爲m*n個子圖，該子圖的位置從位置（a,b）開始畫

#繪製單個特徵下特徵值的分佈情況
plt.subplot2grid((2,3),(0,1))   #第二個子圖從第0行第1列開始畫
#df.feature返回該特徵下的所有值，value_counts()統計每個分類下的人數，'bar'畫柱狀圖
data_train.Pclass.value_counts().plot(kind='bar')   
plt.ylabel(u"人數")
plt.title(u"乘客的等級分佈")

#查看具體特徵 與 最終分類之間的關係
#第三個子圖：根據年齡看獲救人數的分佈情況，畫散點圖
plt.subplot2grid((2,3),(0,2))
#畫散點圖，橫座標爲獲救獲救，縱座標爲年齡值
plt.scatter(data_train.Survived,data_train.Age) 
plt.ylabel(u"年齡")   #y軸標籤
plt.grid(b=True,which='major',axis='y')  #繪製刻度線的網格線，
# Parameters:which(默認爲major); axis(取值爲‘both’,'x','y')表示繪製哪個方向的網格線
plt.title(u"按照年齡看獲救的分佈情況（1表示獲救）")
2、進行特徵和最終分類標籤的關聯統計

#定義圖表對象
fig = plt.figure()
fig.set(alpha=0.2)
#按照等級 分別求獲救人數和未獲救的人數 
Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()#各個等級未獲救的人 數
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()#各個等級獲救的人數
#轉換爲DataFrame對象,使用字典作爲參數。key值爲DataFrame對象的特徵值
df = pd.DataFrame({'獲救'：Survived_1,'未獲救':Survived_0})
#繪製條形圖，堆疊的條形圖
df.plot(kind='bar',stacked=True)
plt.title(u'各個乘客等級的獲救情況')
plt.xlabel(u'乘客的等級')
plt.ylabel(u'人數')
plt.show()
3、篩選符合條件的樣本，作圖分析

#例如：統計女性（Sex）在高等艙(Pclass)的獲救情況(Survived)
fig = plt.figure()
fig.set(alpha=0.65)
plt.title('查看艙等級和性別的獲救情況')

#使用add_subplot()方法添加子圖,返回的是當前繪畫圖像
ax1 = fig.add_subplot(141)#子圖的個數1*4，當前繪製的是第一個子圖
#按照條件篩選
data_train.Survived[data_train.Sex=='female'][data_train.Pclass != 3].value_counts().plot(kind='bar',label='female highclass',color='#FA2479')

#設置x軸上的刻度值，旋轉角度爲0
ax1.set_xticklabels(['獲救','未獲救'],rotation=0)
#設置整個圖表的標籤值
ax1.legend(['女性/高級艙'],loc='best')
plt.show()
4、查看某個特徵對最終的分類結果有無影響

#使用groupby()進行分組
data_train.groupby(['SibSp','Survived']) #先按照SibSp分組，再對每一個分組下按照Survived進行分類
#使用count函數統計分類，最終得到結果是對應的每一行的各個屬性值都是統計得到的人數，所以只取其中一列顯示就可以了
df = pd.DataFrame(g.count()['PassengerId'])
print(df)
5、查看各個屬性之間的相關程度，使用熱度圖

corrmat = train.corr()    #得到特徵之間的相關係數矩陣
#列出與特定的特徵item_cnt_day相關係數由大到小進行排列
#取與item_cnt_day相關係數最大的5個特徵值，由大到小排列的矩陣
cols = corrmat.nlargest(5,'item_cnt_day')['item_cnt_day'].index
#相關係數排列後的值
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)#設置熱度圖的字體
#畫熱度圖
hm = sns.heatmap(cm,cbar=True,annot=True,square=True,fmt='.2f',annot_kws={'size':10},yticklabels=cols.values,xticklabels=cols.values)
plt.show()
三、進行數據處理
1、缺失值

（1）如果缺失的值很多佔樣本總數比例很高，則直接捨棄

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
（2）缺失的樣本個數適中，且該屬性非連續值特徵屬性，則將NaN作爲一個新類別，加到類別特徵中

（3）缺失的樣本個數適中，且屬性值爲連續性特徵值，給定特定範圍，把它離散化，將NaN作爲一個type加到屬性類目中

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