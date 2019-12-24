# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:14:42 2019

@author: 11004076
"""
   
import time
import requests
from bs4 import BeautifulSoup
import pandas
import csv

# 目標URL網址
URL = "https://www.amazon.com/stores/page/51CB0C90-F824-42A4-AB2A-4E59DF5485CB?ingress=0&visitId=c8e1cde7-690f-4f26-88c5-4c6a6f56bb3b&lp_slot=auto-sparkle-hsa-tetris&store_ref=SB_A0251552145E5DJBRDDEB&productGridPageIndex={0}"
       
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
    rows = soup.find_all("li", class_="style__itemOuter__2dxew")
    for row in rows:
        name = row.find("a", class_="style__title__3Z2Cu")
        price = row.find("span", class_="price style__xlarge__1mW1P style__buyPrice__61xrU style__bold__3MCG6")["aria-label"]
        star = row.find("span", class_="a-icon-alt")
        reviews = row.find("span", class_="style__reviewCount__2jU9D")
        
        good= [name, price, star, reviews]
        goods.append(good)
    return goods

def web_scraping_bot(urls):
    all_goods = [["品名","價格","評價","評論數"]]  #巢狀清單
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
    urls = generate_urls(URL, 1, 1)  #爬取1~3頁
    print(urls)
    goods = web_scraping_bot(urls) 
    df = pandas.DataFrame(goods)       #用dataframe列出
    print(df)
    #for good in goods:                #用list列出
    #    print(good)
    #save_to_csv(goods, "Amazon_KB_Rank.csv")