# -*- coding: utf-8 -*-
"""
Created on Tue May 26 14:48:15 2020

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

# 目標URL網址
url = "https://ph.xiapibuy.com/search?keyword={0}&page={1}"
       
def get_urls(url, query, start_page, end_page): 
    urls = []
    for page in range(start_page, end_page+1):
        urls.append(url.format(query, page))    #query帶入url的{0}、page帶入{1}
    return urls

def get_resource(url):
    ua = UserAgent()
    headers = {"user-agent": "Googlebot"}
    return requests.get(url, headers=headers)

def parse_html(html_str):
    return BeautifulSoup(html_str, "lxml")

def get_goods(soup):
    goods = []
    rows = soup.find_all("div",class_="col-xs-2-4 shopee-search-item-result__item")

    for row in rows:  

        try:
            name = row.find("div", class_="_1NoI8_ _16BAGk").text
            #name = row.find("a", class_="name browsinglink").img["alt"]
        except:
            name = None

        try:         
            price = row.find("span", class_="_341bF0").text
            #price = str(row.find("span", class_="p13n-sc-price"))[28:33]
        except:
            price = None     
            
        try:         
            Original_price = row.find("div", class_="_1w9jLI QbH7Ig U90Nhh").text

        except:
            Original_price = None  
            
        try:         
            #都沒東西??
            #discount = row.find("span", class_="percent").text
            discount = str(row.find("span", class_="percent"))[23:25]
        except:
            discount = None  

        try:         
            sold = row.find("div", class_="_18SLBt").text

        except:
            sold = None              

        try:         
            link = "https://ph.xiapibuy.com/" + row.find("a").get('href')

        except:
            link = None   
        
        good= [name, price, Original_price, discount, sold, link]
        goods.append(good)
    return goods

def web_scraping_bot(urls):
    all_goods = [["品名","價格","原價","折扣","售出量", "網址"]]  #巢狀清單
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
    print(get_resource(site).status_code)

    urls = get_urls(url, "genius", 0, 0)
    print(urls)

    goods = web_scraping_bot(urls)
    df = pd.DataFrame(goods)       #用dataframe列出
    print(df)
    #for good in goods:                #用list列出
    #    print(good)
    
    save_to_csv(goods, "Shoppe_Phi.csv")