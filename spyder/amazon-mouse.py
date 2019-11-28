# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:03:33 2019

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
URL = "https://www.amazon.com/Best-Sellers-Computers-Accessories-Computer-Mice/zgbs/pc/11036491/ref=zg_bs_pg_2?_encoding=UTF8&pg={0}"
       
def generate_urls(url, start_page, end_page): #使用參數基底URL、開始和結束頁數來建立URL清單
    urls = []   #爬蟲主程式建立的目標網址清單
    for page in range(start_page, end_page+1):
        urls.append(url.format(page))
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
    rows = soup.find_all("li", class_="zg-item-immersion")
    for row in rows:
        #照理說是直接取.text最快，但不知為何一直出現AttributeError: " 'NoneType' object has no attribute 'text' " 爬文也沒解
        #只好用str()將bs4.element.Tag轉換成字串，再使用 []取部分內容(這個動作叫 slicing)
        rank = row.find("span", class_="zg-badge-text").text
        name = row.find("div", class_="a-section a-spacing-small").img["alt"] #text無法取得，但<img>的alt也有同樣的品名
        star = str(row.find("span", class_="a-icon-alt"))[25:28]  #位置剛好都在25~28之間，所以轉換成str再slicing
        #下面兩行比較特別，先轉換成str，但位置都不同，所以得用find找">"和"<"的位置，再包在slicing中，等同取得">"到"<"之間的字串
        reviews_tag = str(row.find("a", class_="a-size-small a-link-normal"))
        reviews = reviews_tag[reviews_tag.find(">",1)+1 : reviews_tag.find("<",1)]
        price = str(row.find("span", class_="p13n-sc-price"))[28:33]
        
        good= [rank, name, star, reviews, price]
        goods.append(good)
    return goods

def web_scraping_bot(urls):
    all_goods = [["排名","品名","評價","評論數","價格"]]  #巢狀清單
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
    urls = generate_urls(URL, 1, 3)  #爬取1~3頁
    print(urls)
    goods = web_scraping_bot(urls) 
    df = pd.DataFrame(goods)       #用dataframe列出
    #print(df)
    #for good in goods:                #用list列出
    #    print(good)
    
    #save_to_csv(goods, "Amazon_Mouse_Rank.csv")
    
"""
資料分析部分
"""

df.columns = ["rank", "name", "star", "reviews", "price"]  #重新命名欄位名稱
df_pivot = df.pivot_table(values = "name", columns = "star", aggfunc=np.count_nonzero)
print(df_pivot)
    
df_pivot.plot(kind = 'bar')
plt.show()
