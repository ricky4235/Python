# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:21:56 2019

@author: 11004076
"""

#-*-coding:gbk -*-
import urllib2
import lxml.html
import requests

def amazon_price(url, user_agent):
    kv = {'user-agent': user_agent}
    r = requests.get(url, headers = kv)
    r.raise_for_status()
    r.encoding = r.apparent_encoding
    tree = lxml.html.fromstring(r.text.encode("utf-8"))
    price = tree.cssselect("span#priceblock_ourprice")[0]
    return price.text_content().encode("gbk").strip("￥")

if __name__=="__main__": 
    url = "https://www.amazon.cn/dp/B00RG87DUY/"
    user_agent = 'Mozilla/5.0'
    print(amazon_price(url, user_agent))
