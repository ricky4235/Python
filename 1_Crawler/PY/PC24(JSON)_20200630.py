# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 16:11:35 2020

@author: 11004076
"""

import pandas as pd
import requests
import json
import csv

#搜尋網址&換頁
def get_urls(url, query, start_page, end_page): 
    urls = []
    
    for page in range(start_page, end_page+1):
        urls.append(url.format(query, page))    #query帶入url的{0}、page帶入{1}
    print(urls)    
    return urls

def get_goods(raw_data):  

    products = raw_data['prods']
    
    df=pd.DataFrame(products).fillna('null')
    ls=df.values.tolist()
    ls.insert(0,df.columns.tolist())
    
    return ls

def save_to_csv(items, file):
    with open(file, "w+", newline="", encoding="utf_8_sig") as fp:  #utf_8_sig:能讓輸出的csv正確顯示中文(utf_8會有亂碼)
        writer = csv.writer(fp)
        for item in items:
            writer.writerow(item)

if __name__ == "__main__":
    url = "https://ecshweb.pchome.com.tw/search/v3.3/all/results?q={0}&page={1}&sort=sale/dc"
    urls = get_urls(url, "logitech", 0, 3)
    
    goods = [get_goods(requests.get(urls[0]).json())[0]]  #取第0欄(欄位名稱)
    for url in urls:
        res = requests.get(url)
        js = res.json()
        g = get_goods(js)[1:]   #取第1欄之後的資料
        goods.extend(g)  #這裡要用extend(而非append):向列表尾部追加一個列表
    
    save_to_csv(goods, "pc.csv")