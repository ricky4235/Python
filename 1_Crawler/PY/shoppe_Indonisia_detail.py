# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:34:16 2020

@author: 11004076
"""

import pandas as pd
import re, time, requests
import requests
from selenium import webdriver
from bs4 import BeautifulSoup
import csv


def get_soup(url):
    headers = {"user-agent": "Googlebot"}
    res = requests.get(url, headers=headers)
    return BeautifulSoup(res.text, 'lxml')

# 爬取點入網址
def FindLinks(url):
    linklist = []
    soup = get_soup(url)
    links = soup.find_all("div",class_="col-xs-2-4 shopee-search-item-result__item")
    linklist = []
    for link in links:
        k = "https://id.xiapibuy.com/" + link.find("a").get('href')
        linklist.append(k)
    return linklist

# 爬取點入分頁資料
def get_goods(url):
    goods = []
    rows = get_soup(url)

    for row in rows:

        try:
            name = row.find('div', class_="qaNIZv").text
        except:
            name = None

        try:
            price = row.find('div', class_="_3n5NQx").text
        except:
            price = None
            
        try:
            Original_price = row.find("div", class_="_3_ISdg").text
        except:
            Original_price = None
            
        try:
            star = row.find('div', '_3Oj5_n _2z6cUg').text
        except:
            star = None
            
        try:
            sold = row.find('div', '_22sp0A').text
        except:
            sold = None
            
        try:
            stock = row.select_one('._1FzU2Y .items-center div+ div').text
        except:
            stock = None    
        
        
        good= [name, price, Original_price, star, sold, stock]
        goods.append(good)
        
    return goods[1]  #因為不知為何第[0]列都會出現一排None，只好取第[1]列

# 爬取每一個點入頁面
def scraping(urls):
    ndf = []
    for i in FindLinks(urls):
        g = get_goods(i)
        ndf.append(g)
        time.sleep(0.5)
        df_all = pd.DataFrame(ndf,columns = ["name", "price", "Original_price", "star", "sold", "stock"])
    return df_all

# 開始爬蟲
if __name__ == "__main__":
    urls = 'https://id.xiapibuy.com/search?attrId=14478&attrName=Merek&attrVal=genius&facet=12162&noCorrection=true&page=0'
    print(scraping(urls))
    
    save_to_csv(goods, "Shoppe_Indonisia_detail.csv")