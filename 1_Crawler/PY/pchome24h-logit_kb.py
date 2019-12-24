# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:14:42 2019

@author: 11004076
"""
import time
import requests
from bs4 import BeautifulSoup

# 目標URL網址
url = "https://24h.pchome.com.tw/store/DCAHBC"
     
  
def get_resource(url):
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
               "AppleWebKit/537.36 (KHTML, like Gecko)"
               "Chrome/63.0.3239.132 Safari/537.36"}
    return requests.get(url, headers=headers) 

def parse_html(html_str):
    return BeautifulSoup(html_str, "lxml")

def get_goods(soup):
    goods = []
    rows = soup.find_all("dd", id="DCAHBC-A9009*")
    for row in rows:
        name = row.find("span", class_="extra").text
        price = row.find("span", class_="value").text
        
        good= [name, price]
        goods.append(good)
    return goods

def web_scraping_bot(url):
    all_goods = [["品名","評價","樣本數","價格"]]  #巢狀清單
    
    r = get_resource(url)
    if r.status_code == requests.codes.ok:
        soup = parse_html(r.text)
        goods = get_goods(soup)
        all_goods = all_goods + goods
        print("等待5秒鐘...")
        time.sleep(5)
    else:
        print("HTTP請求錯誤...")

    return all_goods

if __name__ == "__main__":
    get_resource(url)
    print(url)
    goods = web_scraping_bot(url) 
    for good in goods:
        print(good)