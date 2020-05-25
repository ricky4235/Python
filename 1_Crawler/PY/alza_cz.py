# -*- coding: utf-8 -*-
"""
Created on Thu May 21 16:49:24 2020

@author: 11004076
"""

import requests
from bs4 import BeautifulSoup
import time
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 目標URL網址

URL = "https://m.alza.cz/genius/v1328-p{0}.htm"

def generate_urls(url, start_page, end_page): #使用參數基底URL、開始和結束頁數來建立URL清單
    urls = []   #爬蟲主程式建立的目標網址清單
    for page in range(start_page, end_page+1):
        urls.append(url.format(page)) #format會讓{帶括號}裡的東西格式化
    return urls

def get_resource(url):
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
               "AppleWebKit/537.36 (KHTML, like Gecko)"
               "Chrome/63.0.3239.132 Safari/537.36"}
    return requests.get(url, headers=headers) 

def parse_html(html_str):
    return BeautifulSoup(html_str, "lxml")

def get_goods(soup):
    goods = []
    rows = soup.find("div", class_="commodity-list list").find_all("a",class_="item ajax")
    #因為網頁常有空值，故需使用try except，不然會遇到None，就整個程式停掉
    #下面會有兩種方法是因為之前不懂，就直接抓標籤，然後硬湊出來，還是把方法留下
    for row in rows:  

        try:
            name = row.find("div", class_="name").text
            #name = row.find("a", class_="name browsinglink").img["alt"]
        except:
            name = None

        try:         
            price = row.find("div", class_="normal").text
            #price = str(row.find("span", class_="p13n-sc-price"))[28:33]
        except:
            price = None     
            
        try:         
            desc = row.find("div", class_="nameext").text

        except:
            desc = None  
        
        good= [name, price, desc]
        goods.append(good)
    return goods

def web_scraping_bot(urls):
    all_goods = [["品名","價格","介紹"]]  #巢狀清單
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
    urls = generate_urls(URL, 1, 6)  #得到1~3頁url的
    print(urls)
    goods = web_scraping_bot(urls)
    df = pd.DataFrame(goods)       #用dataframe列出
    print(df)
    #for good in goods:                #用list列出
    #    print(good)
    
    save_to_csv(goods, "genius_alza.csv")
    