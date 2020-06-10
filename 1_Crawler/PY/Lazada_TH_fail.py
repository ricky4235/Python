# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:43:25 2020

@author: 11004076
"""

import time
import requests
from bs4 import BeautifulSoup
import pandas
import csv

# 目標URL網址
URL = "https://www.lazada.co.th/catalog/?_keyori=ss&from=search_history&page={0}&q=genius%20mouse&spm=a2o4m.home.search.1.3f0a515fKgdUu5&sugg=genius%20mouse_0_1"


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
    rows = soup.find("div", "c1_t2i").find_all("div", class_="c5TXIP")
    
    for row in rows:
        try:
            name = row.find("div", class_="c16H9d").text
        except:
            name = None
        try:
            Discount_price = row.find("span", class_="c13VH6").text
        except:
            Discount_price = None
        try:         
            Original_price = row.find("del", class_="c13VH6").text
        except:
            Original_price = None
        try:         
            Discount = row.find("span", class_="c1hkC1").text
        except:
            Discount = None
        try:
            reviews = row.find("span", class_="c3XbGJ").text
        except:
            reviews = None
        
        good= [name, Discount_price, Original_price, Discount, reviews]
        goods.append(good)
    return goods





def web_scraping_bot(urls):
    all_goods = [["品名", "折價", "原價", "折率", "評論數"]]  #巢狀清單
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
    urls = generate_urls(URL, 1, 1)  #得到1~3頁url的
    print(urls)
    goods = web_scraping_bot(urls) 
    df = pandas.DataFrame(goods)       #用dataframe列出
    print(df)
    #for good in goods:                #用list列出
    #    print(good)
    save_to_csv(goods, "Thailand_Lazada_Genius.csv")