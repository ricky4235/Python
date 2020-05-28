# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:34:16 2020

@author: 11004076
"""

import pandas as pd
import re, time, requests
from selenium import webdriver
from bs4 import BeautifulSoup

#爬取點入網址
def FindLinks(url):
    linklist = []
    driver = webdriver.Chrome()
    driver.get(url)
    soup = BeautifulSoup(driver.page_source)

    links = soup.find_all('a', {'class':'BaseGridItem__content___3LORP'})
    linklist = []
    for link in links:
        k = link.get('href')
        linklist.append(k)
    return linklist

#爬取點入分頁資料
def GetGoods(response):
    soup2 = BeautifulSoup(response.text)
    #url = soup2.find('link')['href']  #多餘的
    df = pd.DataFrame(data = [{'TITLE':soup2.find('span', attrs={'itemprop':'name'}).text,
                               'Price':soup2.find('span', attrs={'class':'has_promo_price'}), 
                               'Activity Price':soup2.find('span',attrs={'class':'price'}).text }],
                               columns = ['TITLE', 'Price', 'Activity Price']) 
    return df

#測試主畫面有無爬到各點入畫面網址
urls = 'https://tw.search.mall.yahoo.com/search/mall/product?kw=%E6%98%86%E7%9B%88&p=%E6%98%86%E7%9B%88&cid=0&clv=0'
FindLinks(urls)

#測試點入分頁有無爬到
url = 'https://tw.mall.yahoo.com/item/GENIUS-%E6%98%86%E7%9B%88-DX-110-USB-%E7%86%B1%E5%8A%9B%E7%B4%85-%E6%9C%89%E7%B7%9A%E5%85%89%E5%AD%B8%E6%BB%91%E9%BC%A0-p0904196550128'
resp = requests.get(url)
GetGoods(resp)

url = 'https://tw.mall.yahoo.com/item/Logitech%E7%BE%85%E6%8A%80-%E7%84%A1%E7%B7%9A%E6%BB%91%E9%BC%A0M235-%E7%81%B0%E3%80%90%E6%84%9B%E8%B2%B7%E3%80%91-p067085113817'
resp = requests.get(url)
GetGoods(resp)

urls = 'https://tw.search.mall.yahoo.com/search/mall/product?cid=hp&clv=0&kw=logitech%E7%84%A1%E7%B7%9A%E6%BB%91%E9%BC%A0&p=logitech%E7%84%A1%E7%B7%9A%E6%BB%91%E9%BC%A0'
ndf = []
for i in FindLinks(urls):
    res = requests.get(i)
    g = GetGoods(res)
    ndf.append(g)
    time.sleep(1)
ndf

#合併資料
df_all = pd.concat(ndf)
df_all