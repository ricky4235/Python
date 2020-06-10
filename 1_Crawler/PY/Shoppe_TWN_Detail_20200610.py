# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 11:49:28 2020

@author: 11004076
"""

"""
遍歷【台灣蝦皮】各商品爬取細節_20200610完整版
1. 先取得搜尋頁數的網址List
2. 進入每個商品頁面
3. 爬取進入頁面之商品細節
"""
import re, time, requests, csv
from bs4 import BeautifulSoup
import csv

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
    print(urls)    
    return urls

# 依序爬取每頁點入網址
def FindLinks(pages):
    linklist = []
    for page in pages:  
        soup = get_soup(page)
        links = soup.find_all("div",class_="col-xs-2-4 shopee-search-item-result__item")
        for link in links:
            k = "https://shopee.tw/" + link.find("a").get('href')
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
            #price = row.find('div', class_="_3n5NQx").text
            price = row.select_one("._3n5NQx").get_text()
        except:
            price = None
            
        try:
            Original_price = row.find("div", class_="_3_ISdg").text
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
            seller_link = "https://shopee.tw" + row.find("a", "btn btn-light btn--s btn--inline btn-light--link Ed2lAD").get('href')
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
            
        try:
            URL = url
        except:
            URL = None
            
        good= [name, price, Original_price, star, reviews, sold, stock, seller, seller_link, seller_from, category, brand, description, URL]
        goods.append(good)
        
    return goods[1]  #因為不知為何第[0]列都會出現一排None，只好取第[1]列

# 將每一個點入頁面的List依序爬取
def scraping(urls):
    all_goods = [["name","price","Original_price","star","reviews","sold","stock","seller","seller_link","seller_from","category","brand","description", "URL"]]
    for idx,i in enumerate(FindLinks(urls)):  #記錄目前進行的迴圈次數，配上總迴圈次數，可做為進度條使用。
        print("Crawing No." + str(idx+1) + " Item in Total:" + str(len(FindLinks(urls))) + "Item")
        
        goods = get_goods(i)
        time.sleep(0.2)
        all_goods.append(goods)
    return all_goods

#存成CSV
def save_to_csv(items, file):
    with open(file, "w+", newline="", encoding="utf_8_sig") as fp:  #utf_8_sig:能讓輸出的csv正確顯示中文(utf_8會有亂碼)
        writer = csv.writer(fp)
        for item in items:
            writer.writerow(item)
    
# 開始爬蟲
if __name__ == "__main__":
    """直接在蝦皮搜尋"""
    url = "https://shopee.tw/search?keyword={0}&page={1}"
    """在電腦配件中搜尋"""
    #url = "https://shopee.tw/search?category=134&keyword={0}&page={1}"
    """直接搜尋品牌"""
    #url = "https://shopee.tw/search?attrId=14478&attrName=Merek&attrVal={0}&page={1}"
    
    urls = get_urls(url, "昆盈", 0, 1)
    
    m = scraping(urls)
    save_to_csv(m, "m.csv")