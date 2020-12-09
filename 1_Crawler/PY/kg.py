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
#============================================================    
from sklearn.ensemble import RandomForestRegressor

def set_Cabin_type(df):
    df.loc[ (df.Cabin.notnull()) ] = "Yes"
    df.loc[ (df.Cabin.isnull()) ] = "No"
    return df

data_train, rfr = set_missing_ages(data_train) 
data_train = set_Cabin_type(data_train) 
data_train


    100 ＜ n ≤ 500 5%
    500 ＜ n ≤ 1000 1%
    n ＞ 2000 
--------

df = df.copy().sample(600)

from statsmodels.formula.api import ols


lm = ols('price ~ C(neighborhood) + C(style)', data=df).fit() 

lm = ols('price ~ area + bedrooms + bathrooms', data=df).fit()
lm.summary()

make_blobs

DESCR  

## Recognizing hand-written digits
('images', (1797L, 8L, 8L))

('data', (1797L, 64L))

('target_names', (10L,))

DESCR

('target', (1797L,))


images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)



n_samples = len(digits.images)


predicted = classifier.predict(data[n_samples // 2:])

expected[:10] :[8 8 4 9 0 8 9 8 1 2]

predicted[:10]:[8 8 4 9 0 8 9 8 1 2]

metrics.confusion_matrix(expected, predicted))

# metrics.plot_confusion_matrix(classifier, X_test, y_test)
def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    import numpy as np
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(digits.target_names))
    plt.xticks(tick_marks, digits.target_names, rotation=45)
    plt.yticks(tick_marks, digits.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    
images_and_predictions = list(
                        zip(digits.images[n_samples // 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()
### sklearn中的make_blobs方法

%matplotlib inline
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

n_train = 20  # samples for training
n_test = 200  # samples for testing
n_averages = 50  # how often to repeat classification
n_features_max = 75  # maximum number of features
step = 4  # step size for the calculation

X, y = generate_data(10, 5)

import pandas as pd
pd.set_option('precision',2)
df=pd.DataFrame(np.hstack([y.reshape(10,1),X]))
df.columns = ['y', 'X0', 'X1', 'X2', 'X2', 'X4']
print(df)

acc_clf1, acc_clf2 = [], []
n_features_range = range(1, n_features_max + 1, step) #n_features_max:最初設定的
for n_features in n_features_range:
    score_clf1, score_clf2 = 0, 0
    for _ in range(n_averages):  #n_averages:最初設定的
        X, y = generate_data(n_train, n_features)

        clf1 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(X, y)
        clf2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None).fit(X, y)

        X, y = generate_data(n_test, n_features)
        score_clf1 += clf1.score(X, y)
        score_clf2 += clf2.score(X, y)

    acc_clf1.append(score_clf1 / n_averages)
    acc_clf2.append(score_clf2 / n_averages)


features_samples_ratio = np.array(n_features_range) / n_train
fig = plt.figure(figsize=(10,6), dpi=300)
plt.plot(features_samples_ratio, acc_clf1, linewidth=2,
         label="Linear Discriminant Analysis with shrinkage", color='r')
plt.plot(features_samples_ratio, acc_clf2, linewidth=2,
         label="Linear Discriminant Analysis", color='g')

plt.xlabel('n_features / n_samples')
plt.ylabel('Classification accuracy')

plt.legend(loc=1, prop={'size': 10})
#plt.suptitle('Linear Discriminant Analysis vs. \
#shrinkage Linear Discriminant Analysis (1 discriminative feature)')
plt.show()



C = 1.0

# Create different classifiers. The logistic regression cannot do
# multiclass out of the box.
classifiers = {'L1 logistic': LogisticRegression(C=C, penalty='l1'),
               'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2'),
               'Linear SVC': SVC(kernel='linear', C=C, probability=True,
                                 random_state=0),
               'L2 logistic (Multinomial)': LogisticRegression(
                C=C, solver='lbfgs', multi_class='multinomial'
                )}

n_classifiers = len(classifiers)

xx, yy = np.meshgrid(np.linspace(1,3,3), np.linspace(4,6,3).T)
Xfull = np.c_[xx.ravel(), yy.ravel()]
print('xx= \n%s\n' % xx)
print('yy= \n%s\n' % yy)
print('xx.ravel()= %s\n' % xx.ravel())
print('Xfull= \n%s' % Xfull)

(三) 測試分類器以及畫出機率分佈圖的選擇
接下來的動作

1. 用迴圈輪過所有的分類器，並計算顯示分類成功率
2. 將Xfull(10000x2矩陣)傳入 classifier.predict_proba()得到probas(10000x3矩陣)。
    這裏的probas矩陣是10000種不同的特徵排列組合所形成的數據，被分類到三種iris 鳶尾花的可能性。
3. 利用reshape((100,100))將10000筆資料排列成二維矩陣，並將機率用影像的方式呈現出來


%matplotlib inline



fig = plt.figure(figsize=(12,12), dpi=300) 


for index, (name, classifier) in enumerate(classifiers.items()):

    classifier.fit(X, y)
    y_pred = classifier.predict(X)
    classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
    print("classif_rate for %s : %f " % (name, classif_rate))

    # View probabilities=
    probas = classifier.predict_proba(Xfull)
    n_classes = np.unique(y_pred).size
    for k in range(n_classes):
        plt.subplot(n_classifiers, n_classes, index * n_classes + k + 1)
        plt.title("Class %d" % k)
        if k == 0:
            plt.ylabel(name)
        imshow_handle = plt.imshow(probas[:, k].reshape((100, 100)),
                                   extent=(3, 9, 1, 5), origin='lower')
        plt.xticks(())
        plt.yticks(())
        idx = (y_pred == k)
        if idx.any():
            plt.scatter(X[idx, 0], X[idx, 1], marker='o', c='k')

ax = plt.axes([0.15, 0.04, 0.7, 0.05])
plt.title("Probability")
plt.colorbar(imshow_handle, cax=ax, orientation='horizontal')


h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Ana.",
         "Quadratic Discriminant Ana."]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    QuadraticDiscriminantAnalysis()]



for ds in datasets:
     X, y = ds

     X = StandardScaler().fit_transform(X)

     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4)

     xx, yy = np.meshgrid

     ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright)

     ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
     
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.rc('font', **{'family': 'Microsoft YaHei, SimHei'})
plt.rcParams['axes.unicode_minus'] = False


for name, clf in zip(names, classifiers):
     clf.fit(X_train, y_train)
     score = clf.score(X_test, y_test)

     # Plot the decision boundary. For that, we will assign a color to each
     # point in the mesh [x_min, m_max]x[y_min, y_max].
     if hasattr(clf, "decision_function"):
         Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
     else:
         Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
 
     # Put the result into a color plot
     Z = Z.reshape(xx.shape)
     ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
     


values = np.arange(0,1.1,0.1)
cmap_values = mpl.cm.get_cmap('red_blue_classes')(values)
import pandas as pd
pd.set_option('precision',2)
df=pd.DataFrame(np.hstack((values.reshape(11,1),cmap_values)))
df.columns = ['Value', 'R', 'G', 'B', 'Alpha']
print(df)

def dataset_fixed_cov():
    '''Generate 2 Gaussians samples with the same covariance matrix'''
    n, dim = 300, 2
    np.random.seed(0)
    C = np.array([[0., -0.23], [0.83, .23]])
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C) + np.array([1, 1])] 
    y = np.hstack((np.zeros(n), np.ones(n))) 
    return X, y


def dataset_cov():

    n, dim = 300, 2
    mp.dataframe()
    np.random.seed(0)
    C = np.array([[0., -1.], [2.5, .7]]) * 2.
    X = np.r_[np.dot(np.random.randn(n, dim), C),
              np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
    y = np.hstack((np.zeros(n), np.ones(n)))
    return X, y

from sklearn.decomposition import PCA

pca = PCA(n_components=len(data))
pca.fit(data)

plt.figure(figsize=(10, 8))

plt(fig((8m,6n))

from imblearn.combine import SMOTETomek
kos = SMOTETomek(random_state=0)
X_kos, y_kos = kos.fit_sample(X_train, y_train)
print(''.format(Counter(y_kos)))


y_train = train['cls'];        
y_test = test['cls']
X_train = train.loc[:, :'X5'];  
X_test = test.loc[:, :'X5']

from imblearn.combine import SMOTETomek

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns

large = 22; med = 16; small = 12
params = {'axes.titlesize': large,
          'legend.fontsize': med,
          'figure.figsize': (16, 10),
          'axes.labelsize': med,
          'axes.titlesize': med,
          'xtick.labelsize': med,
          'ytick.labelsize': med,
          'figure.titlesize': large}
plt.rcParams.update(params)
plt.style.use('seaborn-whitegrid')
sns.set_style("white")

#Removing missing values 
telecom_cust.dropna(inplace = True)
#Remove customer IDs from the data set
df2 = telecom_cust.iloc[:,1:]
#Convertin the predictor variable in a binary numeric variable
df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

#Let's convert all the categorical variables into dummy variables
df_dummies = pd.get_dummies(df2)
df_dummies.head()

x_fearures = np.array([[-1, -2], [-2, -1], [-3, -2], [1, 3], [2, 1], [3, 2]])
y_label = np.array([0, 0, 0, 1, 1, 1])


lr_clf = LogisticRegression()


from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111, projection='3d')

iris_all_class0 = iris_all[iris_all['target']==0].values
iris_all_class1 = iris_all[iris_all['target']==1].values
iris_all_class2 = iris_all[iris_all['target']==2].values
# 'setosa'(0), 'versicolor'(1), 'virginica'(2)
ax.scatter(iris_all_class0[:,0], iris_all_class0[:,1], iris_all_class0[:,2],label='setosa')
ax.scatter(iris_all_class1[:,0], iris_all_class1[:,1], iris_all_class1[:,2],label='versicolor')
ax.scatter(iris_all_class2[:,0], iris_all_class2[:,1], iris_all_class2[:,2],label='virginica')
plt.legend()

k_list = [1, 3, 5, 8, 10, 15]
h = .02
# 创建不同颜色的画布
cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
cmap_bold = ListedColormap(['darkorange', 'c', 'darkblue'])

plt.figure(figsize=(15,14))
# 根据不同的k值进行可视化
for ind,k in enumerate(k_list):
    clf = KNeighborsClassifier(k)
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.subplot(321+ind)  
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)"% k)

plt.show()

plt.figure(figsize=(10,20))
for i, k in enumerate(n_neighbors):

    clf = KNeighborsRegressor(n_neighbors=k, p=2, metric="minkowski")

    y_ = clf.predict(T)
    plt.subplot(6, 1, i + 1)
    plt.scatter(X, y, color='red', label='data')
    plt.plot(T, y_, color='navy', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i)" 
              % (k))

plt.tight_layout()
plt.show()


import numpy as np
import pandas as pd
# kNN分类器
from sklearn.neighbors import KNeighborsClassifier
# kNN数据空值填充
from sklearn.impute import KNNImputer

from sklearn.metrics.pairwise import nan_euclidean_distances
# 交叉验证
from sklearn.model_selection import cross_val_score
# KFlod的函数
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

X = [[1, 2, np.nan], [3, 4, 3], [np.nan, 6, 5], [8, 8, 7]]
imputer = KNNImputer(n_neighbors=2, metric='nan_euclidean')
imputer.fit_transform(X)

nan_euclidean_distances([[np.nan, 6, 5], [3, 4, 3]], [[3, 4, 3], [1, 2, np.nan], [8, 8, 7]])
nan_euclidean_distances([[np.nan, 6, 5], [3, 4, 3]], [[3, 4, 3], [1, 2,
                         

input_file = './horse-colic.csv'
df_data = pd.read_csv(input_file, header=None, na_values='?')


data = df_data.values
ix = [i for i in range(data.shape[1]) if i != 23]
X, y = data[:, ix], data[:, 23]


for i in range(df_data.shape[1]):
    n_miss = df_data[[i]].isnull().sum()
    perc = n_miss / df_data.shape[0] * 100
    
    if n_miss.values[0] > 0:
        print('>Feat: %d, Missing: %d, Missing ratio: (%.2f%%)' % (i, n_miss, perc))


print('KNNImputer before Missing: %d' % sum(np.isnan(X).flatten()))

imputer = KNNImputer()
imputer.fit(X)
Xtrans = imputer.transform(X)

print('KNNImputer after Missing: %d' % sum(np.isnan(Xtrans).flatten()))


import numpy as np
from sklearn.metrics import confusion_matrix
y_pred = [0, 1, 0, 1]
y_true = [0, 1, 1, 0]

chi2
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
warnings.filterwarnings('ignore')

numerical_fea = list(data_train.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numerical_fea,list(data_train.columns)))
label = 'isDefault'
feature = ''
numerical_fea.remove(label)

for data in [data_train, data_test_a]:
    data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')

    data['issueDateDT'] = data['issueDate'].apply(lambda x: x-startdate).dt.days
def logistic()
def find_outliers_by_3segama(data,fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3


import warnings
warnings.filterwarnings('ignore')


Dataframe(X,y) = ()

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
lr = LogisticRegression()
sclf = StackingClassifier(classifiers=[clf1, clf2, clf3], 
                          meta_classifier=lr)

label = ['KNN', 'Random Forest', 'Naive Bayes', 'Stacking Classifier']
clf_list = [clf1, clf2, clf3, sclf]

fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2, 2)
grid = itertools.product([0,1],repeat=2)

clf_cv_mean = []
clf_cv_std = []
for clf, label, grd in zip(clf_list, label, grid):
    f_scores

    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %.2f (+/- %.2f) [%s]" %(scores.mean(), scores.std(), label))
    clf_cv_mean.append(scores.mean())
    clf_cv_std.append(scores.std())

    clf.fit(X, y)
    ax = plt.subplot(gs[grd[0], grd[1]])
    fig = plot_decision_regions(X=X, y=y, clf=clf)
    plt.title(label)

plt.show()
# 種類 2 種以下的類別型欄位轉標籤編碼 (Label Encoding)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le_count = 0

# 檢查每一個 column
for col in app_train:
    if app_train[col].dtype == 'object':
        # 如果只有兩種值的類別型欄位
        if len(list(app_train[col].unique())) <= 2:
            # 就做 Label Encoder
            le.fit(app_train[col])
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            
            # 紀錄有多少個 columns 被標籤編碼過
            le_count += 1
            
# 標籤編碼 (2種類別) 欄位轉 One Hot Encoding            
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)

# 受雇日數為異常值的資料, 另外設一個欄位記錄, 並將異常的日數轉成空值 (np.nan)
app_train['DAYS_EMPLOYED_ANOM'] = app_train["DAYS_EMPLOYED"] == 365243
app_train['DAYS_EMPLOYED'].replace({365243: np.nan}, inplace = True)
app_test['DAYS_EMPLOYED_ANOM'] = app_test["DAYS_EMPLOYED"] == 365243
app_test["DAYS_EMPLOYED"].replace({365243: np.nan}, inplace = True)

# 出生日數 (DAYS_BIRTH) 取絕對值 
app_train['DAYS_BIRTH'] = abs(app_train['DAYS_BIRTH'])
app_test['DAYS_BIRTH'] = abs(app_test['DAYS_BIRTH'])


——有無缺失值，涉及缺失值的處理問題；特徵過多，可能需要採用PCA進行降維等。

（2）數據的各特徵值是什麼類型？是字符串嗎？是離散的還是連續的？

——涉及數據類型轉換問題，可能需要獨熱編碼或者數據格式轉換（如時間格式）

（3）探索數據的統計信息，最值、四分位數、均值、方差

——涉及是否需要歸一化或標準化



3、數據可視化

——可參考matplotlib和seaborn庫筆記

如果特徵不是很多，可以嘗試挨個變量進行觀察（可視化）和處理。

單個變量：連續性、分佈情況，包括偏度（skewness）和峰度（kurtosis）

——是否需要標準化，是否是偏斜數據集。

sns.distplot(train_raw['Sales'], kde=False) #分布图
train_raw['CompetitionOpenSinceMonth'].value_counts(dropna=False) #不同取值的数量
兩個變量：相關性

——一般是和標籤/目標變量的相關性或者目標變量的時間變化趨勢。

sns.barplot(x='DayOfWeek', y='Sales', data=train_raw)  #条形图看相关性
plt.plot(train_raw['Date'],train_raw['Sales'])  #折线图看时间趋势
多個變量：可以簡單回歸一下

——初步判斷一下哪幾個特徵比較多的解釋了目標變量的變化。



二、數據預處理、特徵工程
（一）數據預處理
數據預處理過程是完全應對上一步探索數據集而來，甚至很多時候是邊探索邊順手處理。

1、重複值處理

重複值分為兩類：

（1）重複的值沒有意義，需要去重。

# 判断重复数据
isDuplicated = df.duplicated() # 判断重复数据记录
print (isDuplicated) # 打印输出
# 删除重复值
new_df1 = df.drop_duplicates() # 删除数据记录中所有列值相同的记录
new_df2 = df.drop_duplicates(['col1']) # 删除数据记录中col1值相同的记录
new_df3 = df.drop_duplicates(['col2']) # 删除数据记录中col2值相同的记录
new_df4 = df.drop_duplicates(['col1', 'col2']) # 删除数据记录中指定列（col1/col2）值相同的记录
（2）重複值是有意義的，不需要去重，包括：

假重複：比如有兩個數據點，其某個特徵值是相同的（比如蘋果和小米都是手機）。比如數據庫中某條記錄因為公司在規則上進行了改變，你可以刪除原記錄進行重新添加，也可以保留原記錄（出於以後還會用到，會保留），再添加一條新的，這樣這個數據點也算是重複了。
重複的記錄用於樣本不均衡處理：在開展分類數據建模工作時，樣本不均衡是影響分類模型效果的關鍵因素之一，解決分類方法的一種方法是對少數樣本類別做簡單過採樣，通過隨機過採樣採取簡單複製樣本的策略來增加少數類樣本。經過這種處理方式後，也會在數據記錄中產生相同記錄的多條數據。此時，我們不能對其中重複值執行去重操作。
重複的記錄用於檢測（發現）業務規則中的問題：對於以分析應用為主的數據集而言，存在重複記錄不會直接影響實際運營，畢竟數據集主要用來做分析；但對於事務型的數據而言，重複數據可能意味著重大運營規則問題，尤其當這些重複值出現在與企業經營中金錢相關的業務場景中，例如重複的訂單、重複的充值、重複的預約項、重複的出庫申請等。這些重複的數據記錄通常是由於數據採集、存儲、驗證和審核機制的不完善等問題導致的，會直接反映到前台生產和運營系統。以重複訂單為例，假如前台的提交訂單功能不做唯一性約束，那麼在一次訂單中重複點擊提交訂單按鈕，就會觸發多次重複提交訂單的申請記錄，如果該操作審批通過後，會聯動帶動運營後端的商品分揀、出庫、送貨，如果用戶接收重複商品則會導致重大損失；如果用戶退貨則會增加反向訂單，並影響物流、配送和倉儲相關的各個運營環節，導致運營資源無端消耗、商品損耗增加、倉儲物流成本增加等問題。因此，這些問題必須在前期數據採集和存儲時就通過一定機制解決和避免。如果確實產生了此類問題，那麼數據工作者或運營工作者可以基於這些重複值來發現規則漏洞，並配合相關部門最大限度降低給企業帶來的運營風險。


2、缺失值（Nan）處理

知乎

有缺失值的特徵會給模型帶來極大的噪音，對學習造成較大的干擾。

發現缺失值的時候，首先需要理解缺失背後的原因是什麼：

是數據庫的技術問題還是真正業務的原因導致它缺失？

如果是後者業務原因導致缺失，再來考慮怎麼處理缺失值，處理缺失值的方法有兩類：刪除和填充。

（1）直接刪除法

先看一下這個特徵的缺失率如何：

data.isnull().sum  #统计所有各列各自共有多少缺失值
如果缺失的數量很多，而又沒有證據表明這個特徵很重要，那麼可將這列直接刪除，否則會因為較大的noise對結果造成不良影響。

設定閾值為1%，即保留非缺失值比例大於1%的列（只有缺失比例大於99%才刪除）：

train_B_info = train_B.describe()
meaningful_col = []
for col in train_B_info.columns: 
    if train_B_info.ix[0,col] > train_B.shape[0] * 0.01:
           meaningful_col.append(col)
train_B_1 = train_B[meaningful_col].copy()
# 这里有一个技巧，因为train_B_info的第一行（索引为0）是count，train_B_info.ix[0,col]就是col这一列的count（这一列有多少非缺失值），而train_B.shape[0]即数据点个数（行数）。但是此法只适合数值列，字符串列就不行了，因为字符串不能被describe统计。
如果缺失的數量非常少，而且數據不是時間序列那種必須連續的，那麼可以將缺失值對應的樣本刪除：

test_raw[test_raw['Open'].isnull().values==True] # 因为缺失的较少，可以展示一下该列都有哪几行是缺失的
（2）填充法

通過已有的數據對缺失值進行填補：針對數據的特點，選擇用0、最大值、均值、中位數等填充。缺點是效果一般，因為等於人為增加了噪聲。

# 特殊值填充
train_B_1 = train_B_1.fillna(-999)
# 中位数填充
store_df["CompetitionDistance"].fillna(store_df["CompetitionDistance"].median()) 
# 前向后向填充
data_train.fillna(method='pad')  #用前一个值填充
data_train.fillna(method='bfill')  #用后一个值填充
# 插值填补（通过两点估计中间点的值）
data_train.interpolate() 
# 可以指定方法：interpolate('linear')，线性填充
（3）建立一個模型“預測”缺失的數據

比如KNN, Matrix completion等方法，即用其他特徵做預測模型來算出缺失特徵。缺點是如果缺失變量與其他變量之間沒有太大相關，那麼仍然無法準確預測。

（4）把特徵映射到高維空間（虛擬變量）

比如性別有男、女、缺失三種情況，則映射成3個變量：是否男、是否女、是否缺失。缺點是連續型變量維度過高，計算量很大，而且在數據量不足時會有稀疏性問題。

（5）選擇不受缺失值影響的模型



3、異常值分析

——詳細內容見本專欄異常值檢測筆記。

需注意：異常值檢測有兩種應用，一是作為一個獨立的過程，識別異常的事件或對象；二是作為數據預處理過程的一部分：被看作噪音，識別後還需要進行後續處理（後續處理也見該文）。此處為第二種用途。



4、格式轉換

涉及對字符串類型的數值的處理，可以進行數據類型轉換和顆粒度變換。

數據類型轉換：比如字符串轉換為時間格式；

data1['Date'] = pd.to_datetime(data1['Date'])
顆粒度變換：比如時間序列數據的平滑顯示（按天顯示變按月顯示）；

# 创建新数据集temp，将原数据集的时间列提取出来，作为新数据集的索引。然后对索引进行操作
index = data1['Date']
temp = data1['Sales']
temp.index = index
temp = temp.to_period('M')
temp.groupby(level=0).mean()
# level=0，以索引为依据进行分组




5、離散化、二元化/獨熱編碼（One-Hot Encoding）

如果特徵/屬性的取值本來就是分類的，那麼可以直接使用下面的方法，如果是連續的，則需要先進行離散化，比如年齡屬性的取值是連續的，可以人為分組為小於20歲、 20-60歲、大於60歲三組。

（1）直接替換（映射）

有時只需直接將非數值特徵替換為數值特徵：

比如標籤有兩種類別（"<=50K"和">50K"），可以直接將他們編碼成兩個類0和1。

income = income_raw.replace(['<=50K','>50K'],[0,1])
# 也可以先在外面写成字典形式再传入replace函数
比如屬性值之間存在序（order）的關係，高、中、低，可直接通過連續化編碼為1,0.5,0。

（2）啞變量/虛擬變量（dummy variable）

將分類變量轉換為“啞變量矩陣”或“指標矩陣”：


用pandas.get_dummies()函數：

dummies = pd.get_dummies(user_clu['constellation'])
user_clu = pd.merge(user_clu, dummies,left_index=True,right_index=True)
user_clu = user_clu.drop('constellation',1)
用sklearn preprocessing模塊中的LabelBinarizer函數（標籤二值化函數）：

import numpy as np
from sklearn import preprocessing
# Example labels 示例 labels
labels = np.array([1,5,3,2,1,4,2,1,3])
# Create the encoder 创建编码器
lb = preprocessing.LabelBinarizer()
# Here the encoder finds the classes and assigns one-hot vectors  # 编码器找到类别并分配 one-hot 向量
lb.fit(labels)
# And finally, transform the labels into one-hot encoded vectors # 最后把目标（lables）转换成独热编码的（one-hot encoded）向量
lb.transform(labels)




6、特徵縮放（scaling）：歸一化/標準化/對數轉換

也就是所謂的無量綱化，使不同規格的數據轉換到同一規格。

特徵縮放分兩種：一種是線性縮放，直接把數據壓縮到0-1之間；另一種是如果數據不是正態分佈的，尤其是數據的平均數和中位數相差很大的時候（表示特徵取值非常歪斜），這時通常用一個非線性的縮放。（尤其是對於金融數據）

二者區別：一是對行處理，一是對列處理；一是線性縮放，一是非線性縮放。

不受特徵縮放影響的算法

需要縮放的：logistic回歸、SVM、K均值聚類

不需要縮放的：決策樹、NB

6.1 歸一化（normalization）（centering）

線性縮放，直接把數據壓縮到0-1之間。

（1）sklearn中的最大最小值縮放器（MinMaxScaler）

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])
（2）手動設置：

(x-np.min(x))/(np.max(x)-np.min(x)) 
6.2 標準化（standardization）


對於特徵值的取值不均勻的特徵，可將其取值分佈轉換為均值為0 ，方差為1的正態分佈。因為正態分佈的簡寫也是norm，因此很多地方將數據的標準化同樣叫normalization。

（1）Box-Cox變換/對數轉換

將數據轉換成對數，這樣非常大和非常小的值不會對學習算法產生負面的影響。並且使用對數變換顯著降低了由於異常值所造成的數據范圍異常。

skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
——注意：因為0的對數是沒有定義的，所以我們必須先將數據處理成一個比0稍微大一點的數以成功完成對數轉換。

進行過對數轉換的數據，預測完之後如需恢復：

np.expm1(predictions)
# 如果没有+1就是：np.exp
（2）sklearn中的方法

from sklearn.preprocessing import StandardScaler
StandardScaler().fit_transform(iris.data)


7、樣本不均衡問題：傾斜數據集處理

——傾斜數據集與特徵分佈的偏斜不同

通過data.value_counts()函數可以看目標變量（標籤）的不同取值的數量，如果兩類樣本，一類樣本佔絕大多數，另一類則佔極少數，那麼就是樣本不均衡問題。現實中，比如處理信用卡欺詐問題（交易數據異常檢測），就會面臨這樣一個情況。

——異常值問題，是特徵或標籤都可能異常，這裡專指標籤的取值異常。

解決辦法：下採樣和過採樣。

（1）下採樣（under sample）：

讓兩類樣本同樣少。正例有1萬條，負例有100條，那就從正例中選擇100條和負例搭配。（僅針對訓練過程而言，測試集還是用原來的未下採樣的）

# Number of data points in the minority class
number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace = False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices,random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices,:]
（2）過採樣（over sample）：效果更好（數據一般越多越好）

讓兩類樣本同樣多。人工生成數據，比如生成1萬條負例。

SMOTE（Synthetic Minority Oversampling Technique）算法：

對於每一個少數類中的樣本x，按以下公式生成新樣本：

[公式]

d表示新樣本到原始樣本xi的距離。

# 需事先安装imblearn库
from imblearn.over_sampling import SMOTE
oversampler=SMOTE(random_state=0)
os_features,os_labels=oversampler.fit_sample(features_train,labels_train)


8、假陽性和假陰性這兩種錯誤的重要性不同

比如邊境的安檢會犯兩種錯誤，不是恐怖分子卻當作恐怖分子審訊，是恐怖分子卻當作正常人放行，無疑後者的錯誤更嚴重。

解決方法：賦予不同類樣本不同的權重（或者使用不同的評估方法，見相關筆記）。比如賦予我們更關注的那類樣本更大的權重，使其對最終結果的影響更大。比如一個算法有更高的對貓的識別率，但卻會同時識別一些色情圖片，也就是說它錯誤率低，但錯誤的代價比較大。解決方法是評估指標中給那些色情圖片一個更大的權重，這樣分錯它們的懲罰就會更大。

lr=LogisticRegression(class_weight='balance')
# 如果效果不好，也可自己定义权重项
penalty={0:5,1:1}
lr=LogisticRegression(class_weight=penalty)
7和8的情況，分別對應著評估指標中的查准率和查全率。當然很多情況下，這兩種情況是同時存在的，比如欺詐檢測問題，一方面欺詐樣本很少，是傾斜數據集，另一方面識別欺詐很重要，要賦予不同的權重。

