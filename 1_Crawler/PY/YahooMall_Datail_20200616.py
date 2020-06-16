# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 14:15:01 2020

@author: 11004076
"""

"""
遍歷【Yahoo商城】各商品爬取細節_20200616未完版:走訪兄弟標籤還抓無
1. 先取得搜尋頁數的網址List
2. 進入每個商品頁面
3. 爬取進入頁面之商品細節
"""
import re, time, requests, csv
from bs4 import BeautifulSoup
import csv

#解析(蝦皮headers要用Googlebot)
def get_soup(url):
    #headers = {"user-agent": "Googlebot"}
    res = requests.get(url)
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
        links = soup.find_all('a', {'class':'BaseGridItem__content___3LORP'})
        for link in links:
            k = link.get('href')
            linklist.append(k)
    return linklist

# 爬取點入分頁資料
def get_goods(url):
    goods = []
    rows = get_soup(url)

    for row in rows:

        try:
            name = row.find('div', 'right clearfix').text
        except:
            name = None

        try:
            price = row.find('span', attrs={'class':'has_promo_price'}).text
        except:
            price = None
            
        try:
            Activity_price = row.find('span',attrs={'class':'price'}).text
        except:
           Activity_price = None
            
        try:
            star = row.find('em', 'store').text
        except:
            star = None

        try:
            description = row.find("div", "top").p.span.get_text()
        except:
            description = None
            
        try:
            Product_Number = row.select_one("#ypsiif li span").get_text()
        except:
            Product_Number = None

        try:
            Store_Number = row.find("div", "yui3-g yui3-cssreset cl-mainitem").find_next_sibling("ul").li[1].text
        except:
            Store_Number = None

        try:
            Purchases = row.find("div", "yui3-g yui3-cssreset cl-mainitem").find_next_sibling("ul").li[2].text
        except:
            Purchases = None

        try:
            Sales = row.find("div", "yui3-g yui3-cssreset cl-mainitem").find_next_sibling("ul").li[3].text
        except:
            Sales = None
            
        try:
            URL = url
        except:
            URL = None
            
        good= [name, price, Activity_price, star, description, Product_Number, Store_Number, Purchases, Sales, URL]
        goods.append(good)
        
    return goods[1]  #因為不知為何第[0]列都會出現一排None，只好取第[1]列

# 將每一個點入頁面的List依序爬取
def scraping(urls):
    all_goods = [["商品名稱", "網路價", "活動價", "消費者滿意度", "商品敘述", "商品編號", "店家貨號", "購買人次","銷售件數", "網址"]]
    
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
    """直接在YahooMall搜尋"""
    url = "https://tw.search.mall.yahoo.com/search/mall/product?cid=979311985&clv=4&p={0}&pg={1}&qt=product"
    
    urls = get_urls(url, "昆盈", 0, 1)
    
    m = scraping(urls)
    save_to_csv(m, "m.csv")