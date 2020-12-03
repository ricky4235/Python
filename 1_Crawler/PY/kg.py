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
numerical_fea.remove(label)

for data in [data_train, data_test_a]:
    data['issueDate'] = pd.to_datetime(data['issueDate'],format='%Y-%m-%d')
    startdate = datetime.datetime.strptime('2007-06-01', '%Y-%m-%d')

    data['issueDateDT'] = data['issueDate'].apply(lambda x: x-startdate).dt.days

def find_outliers_by_3segama(data,fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])
    outliers_cut_off = data_std * 3
    lower_rule = data_mean - outliers_cut_off
    upper_rule = data_mean + outliers_cut_off
    data[fea+'_outliers'] = data[fea].apply(lambda x:str('异常值') if x > upper_rule or x < lower_rule else '正常值')
    return data



留出法是直接將數據集D劃分為兩個互斥的集合，其中一個集合作為訓練集S，
另一個作為測試集T。需要注意的是在劃分的時候要盡可能保證數據分佈的一致性，
即避免因數據劃分過程引入額外的偏差而對最終結果產生影響。為了保證數據分佈的一致性，
通常我們採用分層採樣的方式來對數據進行採樣。

Tips：通常，會將數據集D中大約2/3~4/5的樣本作為訓練集，其餘的作為測試集。

②交叉驗證法

k折交叉驗證通常將數據集D分為k份，其中k-1份作為訓練集，剩餘的一份作為測試集，這樣就可以獲得k組訓練/測試集，可以進行k次訓練與測試，最終返回的是k個測試結果的均值。交叉驗證中數據集的劃分依然是依據分層採樣的方式來進行。

對於交叉驗證法，其k值的選取往往決定了評估結果的穩定性和保真性，通常k值選取10。

當k=1的時候，我們稱之為留一法

③自助法

我們每次從數據集D中取一個樣本作為訓練集中的元素，然後把該樣本放回，重複該行為m次，這樣我們就可以得到大小為m的訓練集，在這裡面有的樣本重複出現，有的樣本則沒有出現過，我們把那些沒有出現過的樣本作為測試集。

進行這樣採樣的原因是因為在D中約有36.8%的數據沒有在訓練集中出現過。留出法與交叉驗證法都是使用分層採樣的方式進行數據採樣與劃分，而自助法則是使用有放回重複採樣的方式進行數據採樣

數據集劃分總結

對於數據量充足的時候，通常採用留出法或者k折交叉驗證法來進行訓練/測試集的劃分；
對於數據集小且難以有效劃分訓練/測試集時使用自助法；
對於數據集小且可有效劃分的時候最好使用留一法來進行劃分，因為這種方法最為準確