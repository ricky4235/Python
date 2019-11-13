# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:27:36 2019

@author: 11004076
"""

"""
來源: https://www.twblogs.net/a/5c0d049ebd9eee5e40ba9846
目標：在亞馬遜網站搜索商品，爬取前10頁的商品（名字和價格）

第一步：訪問網站，隱藏爬蟲

亞馬遜對爬蟲限制比較嚴格，修改headers、cookies、代理ip

獲取cookie：f12在console輸入document.cookie()

注意：cookies格式爲字典，{'a':'1','b':'2','c':'3'}

最好自己手動替換，我用記事本替換=爲:就出錯了，因爲cookies內部也有=
"""

import requests

url = 'https://www.amazon.cn/s/field-keywords=spark'

head = {'user-agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}

proxy_id = { "http": "http://61.135.155.82:443"}

cookie = {'session-id':'459-4568418-5692641','ubid-acbcn':'459-5049899-3055220','x-wl-uid':'1AK7YMFc9IzusayDn2fT6Topjz3iAOpR3EeA2UQSqco8fo5PbK2aCpyBA/fdPMfKFqZRHc4IeyuU=','session-token':'OH1wPvfOj6Tylq2nnJcdn5wyxycR/lqyGsGU3+lUtU4mbC0ZD9s8/4Oihd1BlskUQG8zRbLVs9vfWXuiJmnRlDT4x35ircp2uLxOLNYQ4j5pzdFJIqqoZUnhHSJUq2yK80P3LqH8An7faXRCPW9BIqX1wu0WmHlSS9vYAPKA/2SGdV9b//EljYjIVCBjOuR/dKRiYEeGK3li0RJOVz7+vMWg7Rnzbx89QxlbCp0WyquZyVxG6f2mNw=="','session-id-time':'2082787201l'}

r = requests.get(url,headers=head,proxies=proxy_id,cookies=cookie)

r.encoding = r.apparent_encoding

r.text

"""
第二步：解析頁面

通過觀察，商品名稱都放在h2標籤內，商品價值在，取出商品名稱

 

# 解析頁面，採用 bs4 定位

# 獲取商品名稱

"""
import bs4
soup = bs4.BeautifulSoup(r, 'html.parser')

name = soup.find_all('h2')

name

"""
對於價格，因爲每個商品有好幾個價格，所以只爬第一個價格（亞馬遜自營價格）

通過商品名來定位價格，商品和價格不會對應錯

 

通過觀察，商品價格：在商品名稱的父父父弟弟弟節點的span標籤裏

name[0].parent.parent.parent.next_sibling.next_sibling.next_sibling('span')[1].string
"""
 

# 獲取商品價格

namelist = []

pricelist = []

for i in range(len(name)):

    try:
    
        pricelist.append(name[i].parent.parent.parent.next_sibling.next_sibling.next_sibling('span')[1].string)
    
    except:
    
        pricelist.append("null")

"""
第三步：輸出
"""
# 輸出

print("{}\t{}".format("商品名稱", "價格"))

for i in range(len(name)):

    print("{}\t{}".format(name[i].string, pricelist[i]))

 

# 或者輸出爲表

import pandas as pd

shangpin = []

for i in range(len(name)):

    shangpin.append([namelist[i],pricelist[i]])

table = pd.DataFrame(data = shangpin, columns = ['商品名稱','價格'])

table.to_csv('D:/yamasun.csv', index = 0)

"""
加入分頁：
"""
# 獲取頁面

goods = 'spark' # 商品名

pages = 10 #爬多少頁

for n in range(pages):

    page = n+1

url = 'https://www.amazon.cn/s/field-keywords=' + goods + '&page=' + str(page)

 

 

 
"""
最終代碼：
"""
import requests

from bs4 import BeautifulSoup

 

namelist = [] # 商品名稱列表

pricelist = [] # 商品價格列表

shangpin = [] # 商品

# 獲取頁面

goods = 'spark' # 搜索商品名

pages = 10 #爬多少頁

for n in range(pages):

    page = n+1

url = 'https://www.amazon.cn/s/field-keywords=' + goods + '&page=' + str(page)

# 隱藏爬蟲

head = {'user-agent':'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36'}

proxy_id = { "http": "http://61.135.155.82:443"}

cookie = {'session-id':'459-4568418-5692641', 'ubid-acbcn':'459-5049899-3055220','x-wl-uid':'1AK7YMFc9IzusayDn2fT6Topjz3iAOpR3EeA2UQSqco8fo5PbK2aCpyBA/fdPMfKFqZRHc4IeyuU=','session-token':'"OH1wPvfOj6Tylq2nnJcdn5wyxycR/lqyGsGU3+lUtU4mbC0ZD9s8/4Oihd1BlskUQG8zRbLVs9vfWXuiJmnRlDT4x35ircp2uLxOLNYQ4j5pzdFJIqqoZUnhHSJUq2yK80P3LqH8An7faXRCPW9BIqX1wu0WmHlSS9vYAPKA/2SGdV9b//EljYjIVCBjOuR/dKRiYEeGK3li0RJOVz7+vMWg7Rnzbx89QxlbCp0WyquZyVxG6f2mNw=="','csm-hit':'tb:0J5M3DH92ZKHNKA0QBAF+b-0J5M3DH92ZKHNKA0QBAF|1544276572483&adb:adblk_no','session-id-time':'2082787201l'}

r = requests.get(url,headers=head,proxies=proxy_id,cookies=cookie)

# 轉換編碼，apparent_encoding是基於文本推測的編碼

r.encoding = r.apparent_encoding

html = r.text

 

# 解析頁面 (採用 bs4 定位)

# 獲取商品名稱

soup = BeautifulSoup(html, 'html.parser')

name = soup.find_all('h2')

# 獲取商品價格

for i in range(len(name)):

    try:
    
        namelist.append(name[i].string)
        
        pricelist.append(name[i].parent.parent.parent.next_sibling.next_sibling.next_sibling('span')[1].string)
    
        for i in range(len(namelist)):
    
            shangpin.append([namelist[i],pricelist[i]])
    
    except:
    
        pricelist.append("null")

 

# 輸出

# print("{}\t{}".format("商品名稱", "價格"))

# for i in range(len(name)):

# print("{}\t{}".format(namelist[i], pricelist[i]))

 

# 輸出爲表
"""
import pandas as pd

table = pd.DataFrame(data = shangpin, columns = ['商品名稱','價格'])

table.to_csv('D:/yamasun.csv', index = 0)
"""