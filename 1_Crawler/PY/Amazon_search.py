# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 16:37:28 2020

@author: 11004076
"""

import requests
from bs4 import BeautifulSoup
import time
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'https://www.amazon.com/s?k={0}&page={1}&qid=1583476177&ref=sr_pg_{1}'  #{0}:format後的第一個位置、{1}:第二個


def get_urls(url, query, start_page, end_page): #建立可自行輸入"查詢值"和"start~end_page"的url清單
    urls = []
    for page in range(start_page, end_page+1):
        urls.append(url.format(query, page))    #query帶入url的{0}、page帶入{1}
    return urls

def get_resource(url):
    headers = {"user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Mobile Safari/537.36"}
    res = requests.get(url, headers=headers)
    if res.status_code == requests.codes.ok:
        print("請求成功")
        return BeautifulSoup(res.text, "lxml")
    else:
        return print("請求失敗")
    
def get_goods(soup):
    goods = []
    rows = soup.find("span", "rush-component s-latency-cf-section").find("div","s-result-list s-search-results sg-row").find_all("div", "sg-row")
    for row in rows:
        try:
            name = row.find("span", "a-size-medium a-color-base a-text-normal").text
        except:
            name = None
        try:
            price = row.find("span", "a-color-base").text
        except:
            price = None
    
        good = [name, price]
        goods.append(good)
    return goods

def bot(urls):
    all_goods = [["品名", "價格"]]  #巢狀清單
    
    page = 1
    for url in urls:
        print("抓取: 第" + str(page) + "頁 網路資料中...")
        page = page + 1
        soup = get_resource(url)
        goods = get_goods(soup)
        all_goods = all_goods + goods
        print("等待3秒鐘...")
        if soup.find("li", class_="a-disabled a-last"):
            break   #已經沒有下一頁
        time.sleep(3)
    return all_goods

if __name__ == "__main__":
#get_urls
    urls = get_urls(url, 'logitech', 1, 1)
    print(urls)
#get_resource
    goods = bot(urls)
    df = pd.DataFrame(goods)      #用dataframe列出
    print(df)
