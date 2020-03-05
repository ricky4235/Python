# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:34:50 2020

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
URL = "https://www.google.com/search?q=阿根廷經濟&tbm=nws&sxsrf=ALeKk018uycuBxRTE1rnNt99uiz6UP32Pg:1583225149298&ei=PRleXpuuEYyNoATAoK2wAw&start={0}&sa=N&ved=0ahUKEwjbwP2k9f3nAhWMBogKHUBQCzYQ8tMDCFU&biw=2880&bih=1453&dpr=0.67"
#start=0,10,20,30，分別為第1,2,3,4頁，每增加一頁+10

def generate_urls(url, start_page, end_page): #使用參數基底URL、開始和結束頁數來建立URL清單
    urls = []   #爬蟲主程式建立的目標網址清單
    for page in range(start_page, end_page+1, 10): #因為每增加一頁網址中的"start"就+10，所以布長設為10
        urls.append(url.format(page)) #format會讓{帶括號}裡的東西格式化
    return urls

def get_resource(url):
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
               "AppleWebKit/537.36 (KHTML, like Gecko)"
               "Chrome/63.0.3239.132 Safari/537.36"}
    return requests.get(url, headers=headers) 

def parse_html(html_str):
    return BeautifulSoup(html_str, "lxml")

def get_news(soup):
    news = []
    rows = soup.find("div", class_="bkWMgd").find_all("div", class_="g")

    for row in rows:
        try:
            title = row.find("a", class_="l lLrAF").text  #不知為何這個抓出來都空白
            #name = row.find("a", class_="style__title__3Z2Cu").img["alt"]
        except:
            title = None
        try:
            medium = row.find("span", class_="xQ82C e8fRJf").text
            #star = str(row.find("span", class_="a-icon-alt"))[25:28]
        except:
            medium = None
        try:
            date = row.find("span", class_="f nsa fwzPFf").text
        except:
            date = None
        try:
            news_url = row.find("a", class_="l lLrAF")["href"]
        except:
            news_url = None
        
        new= [title, medium, date, news_url]
        news.append(new)
    return news

def web_scraping_bot(urls):
    all_news = [["標題","媒體","發布日","網址"]]  #巢狀清單
    page = 1
    
    for url in urls:
        print("抓取: 第" + str(page) + "頁 網路資料中...")
        page = page + 1
        r = get_resource(url)
        if r.status_code == requests.codes.ok:
            soup = parse_html(r.text)
            news = get_news(soup)
            all_news = all_news + news
            print("等待2秒鐘...")
            if soup.find("li", class_="a-disabled a-last"):
                break   #已經沒有下一頁
            time.sleep(2) 
        else:
            print("HTTP請求錯誤...")

    return all_news

def save_to_csv(items, file):
    with open(file, "w+", newline="", encoding="utf_8_sig") as fp:  #utf_8_sig:能讓輸出的csv正確顯示中文(utf_8會有亂碼)
        writer = csv.writer(fp)
        for item in items:
            writer.writerow(item)

if __name__ == "__main__":
    urls = generate_urls(URL, 0, 20)  #得到1~3頁url的[0,10,20]=[第一頁、第二頁、第三頁]
    print(urls)
    news = web_scraping_bot(urls)
    df = pd.DataFrame(news)       #用dataframe列出
    print(df)
    for new in news:                #用list列出
        print(new)
    
    save_to_csv(news, "Contry_Economic_News.csv")