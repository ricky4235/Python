# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:14:42 2019

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
#URL = "https://www.amazon.com/stores/page/D3DB9423-BA6A-4ADC-8E17-C788DD52A2F9?ingress=0&visitId=c7e247ba-88d9-4be7-b52d-77dee1317f21&lp_slot=auto-sparkle-hsa-tetris&store_ref=SB_A10245962MPT6BZPXZHS0&pg=1" #Speaker
URL = "https://www.amazon.com/stores/page/77252D0B-5E49-4F7B-A9D2-1B0B5338FE1C?ingress=0&visitId=189f60dd-4be6-459c-933a-30a59c011fd2&lp_slot=auto-sparkle-hsa-tetris&store_ref=SB_A0251552145E5DJBRDDEB" #Headset
#url_no_pg = "https://www.amazon.com/stores/page/8C991D1E-8C1A-4A94-8486-F251BDCE6E80?ingress=0&visitId=189f60dd-4be6-459c-933a-30a59c011fd2&lp_slot=auto-sparkle-hsa-tetris&store_ref=SB_A0251552145E5DJBRDDEB" #Keyboard
#url_no_pg = "https://www.amazon.com/stores/page/51CB0C90-F824-42A4-AB2A-4E59DF5485CB?ingress=0&visitId=189f60dd-4be6-459c-933a-30a59c011fd2&lp_slot=auto-sparkle-hsa-tetris&store_ref=SB_A0251552145E5DJBRDDEB" #Mouse


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
    rows = soup.find_all("li", class_="style__itemOuter__2dxew")

    for row in rows:

        name = row.find("a", class_="style__title__3Z2Cu").text  
        star = row.find("span", class_="a-icon-alt").text
        reviews = row.find("span", class_="style__reviewCount__2jU9D").text
        price_int = row.find("span", class_="style__whole__3EZEk").text
        price_decimal = row.find("span", class_="style__fractional__iPo3s").text
        price= price_int + "." + price_decimal

        good= [name, star, reviews, price]
        goods.append(good)
    return goods

def web_scraping_bot(urls):
    all_goods = [["品名","評價","評論數","價格"]]  #巢狀清單
    page = 1

"""

import requests
from bs4 import BeautifulSoup
import time

import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
  
"""     
    for url in urls:
        print("crawling: " + str(page) + "page...")
        page = page + 1
        r = get_resource(url)
        if r.status_code == requests.codes.ok:
            soup = parse_html(r.text)
            goods = get_goods(soup)
            all_goods = all_goods + goods
            print("wait 3 second...")
            if soup.find("li", class_="a-disabled a-last"):
                break   #已經沒有下一頁
            time.sleep(3)
        else:
            print("HTTP Request Error...")

    return all_goods

def save_to_csv(items, file):
    with open(file, "w+", newline="", encoding="utf_8_sig") as fp:  #utf_8_sig:能讓輸出的csv正確顯示中文(utf_8會有亂碼)
        writer = csv.writer(fp)
        for item in items:
            writer.writerow(item)
1137.961.624
if __name__ == "__main__":
    urls = generate_urls(URL, 1, 3)  #得到1~3頁url的
    print(urls)
    goods = web_scraping_bot(urls) 
    df = pd.DataFrame(goods)       #用dataframe列出
    print(df)
    for good in goods:                #用list列出
        print(good)  
    save_to_csv(goods, "Amazon_Genius.csv")