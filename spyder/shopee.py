# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 13:53:22 2019

@author: 11004076
"""

import requests
from bs4 import BeautifulSoup

def shopee_scraper(keyword,n_page=0,used=False,new=False):
    '''
    參數說明:
        keyword: 商品名稱關鍵字
        n_page: 第幾頁(每頁有50個商品)
        used: 是否為二手商品?
        new: 是否為新商品?
    '''
    url = f'https://shopee.tw/search?keyword={keyword}&page={n_page}&sortBy=relevancy'
    if used:
        url += '&newItem=true'
    if new:
        url += '&usedItem=true'
    
    headers = {
        'User-Agent': 'Googlebot',
        'From': 'YOUR EMAIL ADDRESS'
    }
    
    r = requests.get(url,headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')
    contents = soup.find_all("div", class_="_1NoI8_ _2gr36I")
    prices = soup.find_all("span", class_="_341bF0")
    all_items = soup.find_all("div", class_="col-xs-2-4 shopee-search-item-result__item")
    links = [i.find('a').get('href') for i in all_items]
    
    for c, p, l in zip(contents, prices, links):
        print(c.contents[0])
        print(p.contents[0])
        print('https://shopee.tw/'+l)
        print('*---------------------------------*')
        
if __name__ == "__main__":
    g = shopee_scraper("昆盈", n_page=1)
    print(g)