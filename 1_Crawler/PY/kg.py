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


#%matplotlib inline  

np.random.seed(2) #

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop #
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau


sns.set(style='white', context='notebook', palette='deep') 




接下來，我們將Colab中的batch size改為256，對模型進行兩次疊代訓練。
上述的改變導致平均運行時間變成了18:38分鐘。將batch size改為64，同樣進行兩次疊代訓練，此時得到的平均運行時間為18:14分鐘。這表示，當batch size大於16的時候，Colab能夠縮減運行的時間。

儘管如此，對於本節中的任務而言，較小的batch size並不是一個值得深究的大問題，有關參數設置的討論，可以參見這篇文章（https://arxiv.org/abs/1804.07612）。

當我將Colab上的batch size設為256，然後開始訓練模型時，Colab拋出了一個警告，其中寫道：我正在使用的GPU具有11.17GB的顯存。具體如下圖所示。

