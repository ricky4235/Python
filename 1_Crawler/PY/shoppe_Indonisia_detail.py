# -*- coding: utf-8 -*-
"""
Created on Thu May 28 15:34:16 2020

@author: 11004076
"""

import pandas as pd
import re, time, requests
import requests
from selenium import webdriver
from bs4 import BeautifulSoup

#解析(蝦皮headers要用Googlebot)
def get_soup(url):
    headers = {"user-agent": "Googlebot"}
    res = requests.get(url, headers=headers)
    return BeautifulSoup(res.text, 'lxml')

#搜尋網址&換頁
def get_urls(url, query, start_page, end_page): 
    urls = []
    for page in range(start_page, end_page+1):
        urls.append(url.format(query, page))    #query帶入url的{0}、page帶入{1}
    return urls

# 依序爬取每頁點入網址
def FindLinks(pages):
    linklist = []
    for page in pages:
        
        soup = get_soup(page)
        links = soup.find_all("div",class_="col-xs-2-4 shopee-search-item-result__item")
        for link in links:
            k = "https://id.xiapibuy.com/" + link.find("a").get('href')
            linklist.append(k)
    return linklist

# 爬取點入分頁資料
def get_goods(url):
    goods = []
    rows = get_soup(url)

    for row in rows:

        try:
            name = row.find('div', class_="qaNIZv").text
        except:
            name = None

        try:
            #price = row.find('div', class_="_3n5NQx").text.replace("Rp","")
            price = row.select_one("._3n5NQx").get_text().replace("Rp","")
        except:
            price = None
            
        try:
            Original_price = row.find("div", class_="_3_ISdg").text.replace("Rp","")
        except:
            Original_price = None
            
        try:
            star = row.find('div', '_3Oj5_n _2z6cUg').text
        except:
            star = None

        try:
            #reviews = row.select("._3Oj5_n")[1].get_text()
            reviews = row.find_all('div', '_3Oj5_n')[1].text
        except:
            reviews = None
            
        try:
            sold = row.find('div', '_22sp0A').text
        except:
            sold = None
            
        try:
            #stock = row.select("._1FzU2Y .items-center div+ div")[0].get_text()
            stock = row.find("div", "flex items-center _1FzU2Y").get_text()
        except:
            stock = None
        
        try:
            seller = row.select_one("._3Lybjn").get_text()
        except:
            seller = None
            
        try:
            seller_link = "https://id.xiapibuy.com" + row.find("a", "btn btn-light btn--s btn--inline btn-light--link Ed2lAD").get('href')
        except:
            seller_link = None
            
        try:
            seller_from = row.select(".kIo6pj")[-1].div.get_text()
            #seller_from = row.find_all("kIo6pj")[-1].text
        except:
            seller_from = None
            
        try:
            category = row.select_one(".kIo6pj").get_text().replace("Kategori", "")
        except:
            category = None

        try:
            brand = row.select_one("._2H-513").get_text()
        except:
            brand = None

        try:
            description = row.find("div", "_2u0jt9").get_text()
        except:
            description = None
            
        good= [name, price, Original_price, star, reviews, sold, stock, seller, seller_link, seller_from, category, brand, description]
        goods.append(good)
        
    return goods[1]  #因為不知為何第[0]列都會出現一排None，只好取第[1]列

# 爬取每一個點入頁面
def scraping(urls):
    ndf = []
    for i in FindLinks(urls):
        g = get_goods(i)
        ndf.append(g)
        time.sleep(0.5)
        df_all = pd.DataFrame(ndf,columns = ["name","price","Original_price","star","reviews","sold","stock","seller","seller_link","seller_from","category","brand","description"])
    return df_all

# 開始爬蟲
if __name__ == "__main__":
    """直接在蝦皮搜尋"""
    #url = "https://id.xiapibuy.com/search?keyword={0}&page={1}"
    """在電腦配件中搜尋"""
    #url = "https://id.xiapibuy.com/search?category=134&keyword={0}&page={1}"
    """直接搜尋品牌"""
    url = "https://id.xiapibuy.com/search?attrId=14478&attrName=Merek&attrVal={0}&page={1}"
    
    urls = get_urls(url, "genius", 6, 11)
    
    pd_links = pd.DataFrame(FindLinks(urls),columns = ["link"])
    #要合併分別爬的商品資料&網址list，只找到join方法(但皆要先轉成DataFrame)
    #且要同時跑才會對應到一樣順序的資料
    m = scraping(urls).join(pd_links)  
    m.to_csv("Shoppe_Indonisia2.csv")