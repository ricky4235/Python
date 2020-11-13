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
    df.loc[ (df.Cabin.notnull()), 'Cabin' ] = "Yes"
    df.loc[ (df.Cabin.isnull()), 'Cabin' ] = "No"
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
    '''Generate 2 Gaussians samples with different covariance matrices'''
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




from imblearn.combine import SMOTETomek
kos = SMOTETomek(random_state=0)  # 综合采样
X_kos, y_kos = kos.fit_sample(X_train, y_train)
print('综合采样后，训练集 y_kos 中的分类情况：{}'.format(Counter(y_kos)))
不难两种过采样方法都将原来 y_train 中的占比少的分类 1 提到了与 0 数量一致的情况
但因为综合采样在过采样后会使用欠采样，所以数量会稍微少一点点