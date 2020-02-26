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
#url_no_pg = "https://www.amazon.com/Best-Sellers-Computers-Accessories-Computer-Mice/zgbs/pc/11036491/ref=zg_bs_pg_2?_encoding=UTF8" #美國
#url_no_pg = "https://www.amazon.cn/gp/bestsellers/pc/1454004071/ref=zg_bs_nav_pc_3_888467051" #中國
#url_no_pg = "https://www.amazon.ae/gp/bestsellers/computers/12050386031/ref=zg_bs_nav_4_12050385031" #阿拉伯
url_no_pg = "https://www.amazon.com.br/gp/bestsellers/computers/16364917011/ref=zg_bs_nav_computers_4_16364919011" #巴西
URL = url_no_pg +"&pg={0}"  #加入換頁字串

       
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
    rows = soup.find_all("li", class_="zg-item-immersion")
    #因為網頁常有空值，故需使用try except，不然會遇到None，就整個程式停掉
    #下面會有兩種方法是因為之前不懂，就直接抓標籤，然後硬湊出來，還是把方法留下
    for row in rows:  
        rank = row.find("span", class_="zg-badge-text").text #因為這不會有None，不需使用try except
        try:
            #name = row.find("div", class_="p13n-sc-truncated").text  #不知為何這個抓出來都空白
            name = row.find("div", class_="a-section a-spacing-small").img["alt"]
        except:
            name = None
        try:
            star = row.find("span", class_="a-icon-alt").text
            #star = str(row.find("span", class_="a-icon-alt"))[25:28]
        except:
            star = None
        try:
            reviews = row.find("a", class_="a-size-small a-link-normal").text
            #下面兩行比較特別，先轉換成str，但位置都不同，所以得用find找">"和"<"的位置，再包在slicing中，等同取得">"到"<"之間的字串
            #reviews_tag = str(row.find("a", class_="a-size-small a-link-normal"))
            #reviews = reviews_tag[reviews_tag.find(">",1)+1 : reviews_tag.find("<",1)]
        except:
            reviews = None
        try:         
            price = row.find("span", class_="p13n-sc-price").text
            #price = str(row.find("span", class_="p13n-sc-price"))[28:33]
        except:
            price = None            
        
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
    print(df)
    for good in goods:                #用list列出
        print(good)
    
    save_to_csv(goods, "Amazon_Mouse_Rank.csv")
    
"""
資料分析部分
"""

df.columns = ["rank", "name", "star", "reviews", "price"]  #重新命名欄位名稱
df_pivot = df.pivot_table(values = "name", columns = "star", aggfunc=np.count_nonzero)
print(df_pivot)
    
df_pivot.plot(kind = 'bar')
plt.show()
