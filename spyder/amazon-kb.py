# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 12:14:42 2019

@author: 11004076
"""
   
import time
import requests
from bs4 import BeautifulSoup

# 目標URL網址
URL = "https://www.amazon.com/Best-Sellers-Computers-Accessories-Computer-Keyboards/zgbs/pc/12879431/ref=zg_bs_nav_pc_4_11036491_encoding=UTF8&pg={0}"
       
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
    rows = soup.find_all("li", class_="zg-item-immersion")  #a-fixed-left-grid-col a-col-right  #a-section a-spacing-none aok-relative
    for row in rows:
        name = row.find("div", class_="p13n-sc-truncated")
        star = row.find("span", class_="a-icon-alt").get_text()
        sample = row.find("a", class_="a-size-small a-link-normal").text
        price = row.find("span", class_="p13n-sc-price").text
        
        good= [name, star, sample, price]
        goods.append(good)
    return goods

def web_scraping_bot(urls):
    all_goods = [["品名","評價","樣本數","價格"]]  #巢狀清單
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
            if soup.find("li", class_="nexttxt disabled"):
                break   #已經沒有下一頁
            time.sleep(5) 
        else:
            print("HTTP請求錯誤...")

    return all_goods

if __name__ == "__main__":
    urls = generate_urls(URL, 1, 1) 
    # print(urls)
    goods = web_scraping_bot(urls) 
    for good in goods:
        print(good)