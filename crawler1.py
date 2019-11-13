#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 17:18:51 2019

@author: ricky
"""

from bs4 import BeautifulSoup  #從HTML/HML中提取資料的包
import requests    #處理網頁(HTTP)請求的包
google=requests.get('http://google.com').content  #取得網站原始碼
print(google)

soup=BeautifulSoup(google,'html.parser')  #解析
links=soup.findAll('a')     #抓網站原始碼的<a>
print(links)

for link in links:          #在每個<a>中迴圈，只抓其字串
    print(link.string)
    
print(soup.title)   #獲取網站名稱
print(soup.body.div.attrs)  #一步一步的抓下層< >