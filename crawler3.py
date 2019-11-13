# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 17:52:35 2019

@author: 11004076
"""

# 引入撰寫爬蟲所需套件
from bs4 import BeautifulSoup  #從HTML/HML中提取資料的包
import requests    #處理網頁(HTTP)請求的包

#定義網址
url='https://www.pcstore.com.tw/geniustw1985/'
res=requests.get(url).content  #取得網站原始碼
#print(google)

#向網址要回網頁原始碼，並透過 BeautifulSoup 解析
soup=BeautifulSoup(res,'html.parser')  #解析


links=soup.find_all('td')     #抓網站原始碼的<標籤>
for link in links:          #在每個<span>中迴圈，只抓其字串
    print(link.span)

print(soup.title)   #獲取網站名稱
print(soup.body.div.attrs)  #一步一步的抓下層<標籤>