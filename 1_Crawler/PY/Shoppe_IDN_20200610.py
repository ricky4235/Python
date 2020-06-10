# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:20:56 2020

@author: 11004076
"""

"""
爬取【印尼蝦皮】搜尋頁面簡易商品資料_20200610完整版
1. 使用針對蝦皮的headers = {"user-agent": "Googlebot"}取得解析網頁資料
2. 先取得搜尋頁數的網址List
3. 直接爬取網址List之商品資料
"""
import requests
from bs4 import BeautifulSoup
import time
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fake_useragent import UserAgent


def get_urls(url, query, start_page, end_page): 
    urls = []
    for page in range(start_page, end_page+1):
        urls.append(url.format(query, page-1))    #query帶入url的{0}、page帶入{1}
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
        except:
            name = None

        try:
            price = row.find("span", class_="_341bF0").text
        except:
            price = None
            
        try:
            Original_price = row.find("div", class_="_1w9jLI QbH7Ig U90Nhh").text.replace("Rp","")
        except:
            Original_price = None

        try:
            sold = row.find("div", class_="_18SLBt").text.replace(" Terjual","")
            #假如字串有包含RB，將RB取代掉，並將字串轉為浮點數，再乘以1000
            if "RB" in sold:
                sold = sold.replace("RB","")
                sold = float(sold)*1000
        except:
            sold = None

        try:
            link = "https://id.xiapibuy.com/" + row.find("a").get('href')
        except:
            link = None
        
        good= [name, price, Original_price, sold, link]
        goods.append(good)
    return goods

def web_scraping_bot(urls):
    all_goods = [["品名","價格(單位:千)","原價(單位:千)","售出量", "網址"]]  #巢狀清單
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
            
            #當目前頁數=所有頁數時，跳出迴圈
            now_page = soup.find("span", class_="shopee-mini-page-controller__current").text
            all_page = soup.find("span", class_="shopee-mini-page-controller__total").text
            if now_page == all_page:
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
    # 目標URL網址
    """直接在蝦皮搜尋"""
    #url = "https://id.xiapibuy.com/search?keyword={0}&page={1}"
    """在電腦配件中搜尋"""
    #url = "https://id.xiapibuy.com/search?category=134&keyword={0}&page={1}"
    """直接搜尋品牌"""
    url = "https://id.xiapibuy.com/search?attrId=14478&attrName=Merek&attrVal={0}&page={1}"

    print(get_resource(url).status_code)

    urls = get_urls(url, "genius", 1, 3)
    print(urls)

    goods = web_scraping_bot(urls)
    df = pd.DataFrame(goods)       #用dataframe列出
    print(df)
    
    save_to_csv(goods, "x.csv")