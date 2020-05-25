# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:12:27 2020

@author: 11004076
"""

import requests
from bs4 import BeautifulSoup
import time
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fake_useragent import UserAgent
ua = UserAgent()
for i in range(5):
    print(ua.random)

# 目標URL網址

URL = "https://www.mvideo.ru/product-list-page-cls?category_id=cat2_cis_0000000{0}&_requestid=737458&region_id=1&q=genius&params=0&limit=12&offset=0"
       
def get_urls(url, cat): #建立可自行輸入"查詢值"和"start~end_page"的url清單
    urls = []
    for cat in cats:
        urls.append(url.format(cat))    #query帶入url的{0}、page帶入{1}
    return urls

def get_resource(url):
    ua = UserAgent()
    headers = {"user-agent": ua.random}
    return requests.get(url, headers=headers, allow_redirects=False) 
#allow_redirects=False:為了避免status code顯示200，但事實上中途有被重新導向，而誤解為有正確的拜訪網站
#加入後直接出現302，現在我們可以避免中途明明被重新導向過，卻顯示一切正常(200)


def parse_html(html_str):
    return BeautifulSoup(html_str, "lxml")

def get_goods(soup):
    goods = []
    rows = soup.find_all("div",class_="c-product-tile sel-product-tile-main")
    #因為網頁常有空值，故需使用try except，不然會遇到None，就整個程式停掉
    #下面會有兩種方法是因為之前不懂，就直接抓標籤，然後硬湊出來，還是把方法留下
    for row in rows:  

        try:
            name = row.find("a", class_="sel-product-tile-title").text
            #name = row.find("a", class_="name browsinglink").img["alt"]
        except:
            name = None

        try:         
            price = row.find("div", class_="c-pdp-price__current").text
            #price = str(row.find("span", class_="p13n-sc-price"))[28:33]
        except:
            price = None     
            
        try:         
            desc = row.find("a", class_="sel-product-tile-title").text

        except:
            desc = None  
        
        good= [name, price, desc]
        goods.append(good)
    return goods

def web_scraping_bot(urls):
    all_goods = [["品名","價格","產品規格"]]  #巢狀清單
    page = 1
    
    for url in urls:
        print("抓取: 第" + str(page) + "頁 網路資料中...")
        page = page + 1
        r = get_resource(url)
        if r.status_code == requests.codes.ok:
            soup = parse_html(r.text)
            goods = get_goods(soup)
            all_goods = all_goods + goods
            print("等待5秒鐘...")
            if soup.find("li", class_="a-disabled a-last"):
                break   #已經沒有下一頁
            time.sleep(5) 
        else:
            print("HTTP請求錯誤...")

    return all_goods

def save_to_csv(items, file):
    with open(file, "w+", newline="", encoding="utf_8_sig") as fp:  #utf_8_sig:能讓輸出的csv正確顯示中文(utf_8會有亂碼)
        writer = csv.writer(fp)
        for item in items:
            writer.writerow(item)

if __name__ == "__main__":
    #print(get_resource(URL).history)
    print(get_resource(URL).status_code)

    """
    cats = ["294", "295", "378"]
    urls = get_urls(URL, cats)
    print(urls)

    goods = web_scraping_bot(urls)
    df = pd.DataFrame(goods)       #用dataframe列出
    print(df)
    #for good in goods:                #用list列出
    #    print(good)
    
    save_to_csv(goods, "genius_mvideo.csv")
"""