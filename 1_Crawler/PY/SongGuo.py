# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 13:58:19 2019

@author: 11004076
"""

"""
使用requests_html
進入每一個商品頁面抓取完整細項
"""

# 載入相關套件
import requests
from requests_html import HTML
import pandas as pd
import re

# 輸入爬蟲網址與使用者資訊
url = 'https://www.pcone.com.tw/product/'
# 男生服飾
info = '327'

# 加入使用者資訊(如使用什麼瀏覽器、作業系統...等資訊)模擬真實瀏覽網頁的情況
headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}

# 找出所有的商品連結
resp = requests.get(url + info, headers=headers)
html = HTML(html=resp.text)
a = html.find('a.product-list-item')


# 定義ProdList函數，輸入商品分類編號，輸出該分類下所有商品連結
def ProdList(info):
    resp = requests.get(url + str(info), headers=headers)
    html = HTML(html=resp.text)
    return(html.find('a.product-list-item'))

# 定義Crawl_SongGuo函數，輸入商品網址，輸出該商品的各項屬性
def Crawl_SongGuo(info):
    resp = requests.get('https://www.pcone.com.tw/product/info/' + re.search(r'\d{12}',str(info)).group(), headers=headers)
    html = HTML(html=resp.text)
    print(re.search(r'\d{12}',str(info)).group())
    return(pd.DataFrame(
            data=[{
                '店家名稱':html.find('a.store-name',first = True).text,
                '店家商品數量':html.find('div.store-val',first = False)[0].attrs['data-val'],
                '店家評價':html.find('div.store-val',first = False)[1].attrs['data-val'],
                '店家出貨天數':html.find('div.store-val',first = False)[2].attrs['data-val'],
                '店家回覆率':html.find('div.store-val',first = False)[3].attrs['data-val'],
                '產品名稱':html.find('h1.product-name',first = True).text,
                '特價':html.find('span.bind-lowest-price.discount',first = True).text,
                '原價':html.find('span.original',first = True).text,
                '折數':html.find('span.bind-discount-number.discount-number',first = True).text,
                '商品評分':html.find('span.count > span',first = False)[0].text,
                '評價人數':html.find('span.count > span',first = False)[1].text,
                '收藏人數':html.find('div.count',first = False)[0].text,
                '提問人數':html.find('div.count',first = False)[1].text,
                '商品分類':html.find('div.breadcrumbs-set',first = True).text,
                '商品標籤':html.find('div.tags',first = True).text,
                '連結':'https://www.pcone.com.tw/product/info/' + re.search(r'\d{12}',str(info)).group()}],
            columns = ['店家名稱', '店家商品數量', '店家評價', '店家出貨天數', '店家回覆率',  '產品名稱', '特價', '原價', '折數',
                       '商品評分', '評價人數', '收藏人數','提問人數', '商品分類', '商品標籤', '連結']))

# 組合以上兩個函數，輸入商品分類的編號，即自動爬出所有商品的屬性，並將資料存在df中
prodlist = ProdList(327)
df = pd.DataFrame()
for i in range(len(prodlist)):
    df = df.append(Crawl_SongGuo(prodlist[i]), ignore_index=True)

# 檢視抓下來的資料
df

# 將df轉成excel並存在桌面上
df.to_excel('SongGuo1.xlsx')