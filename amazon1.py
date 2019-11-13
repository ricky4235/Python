# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 16:35:50 2019

@author: 11004076
"""

"""
https://mjzj.com/article/35205
"""

import requests  # 导入requests
from bs4 import BeautifulSoup  # 从bs4中导入BeautifulSoup
import csv
url = 'https://www.amazon.com/s/keywords=iphone'
response = requests.get(url)
response_soup = BeautifulSoup(response.text, 'html.parser')
result_list = response_soup.find('div', class_='s-results-list s-search-results sg-row').find_all("div")
for li in result_list:  
    info_list = []
    try:
        price = li.find('span', 'a-offscreen').text
    except:
        price = li.find('span', 'a-size-base a-color-base').text
    asin = li['data-asin']
    star = li.find('span', 'a-icon-alt').text
    print(asin)
    print(price)
    print(star)
    info_list.append(asin)
    info_list.append(price)
    info_list.append(star)
    csvFile = open('./iphone.csv', 'a', newline='')
    writer = csv.writer(csvFile)
    writer.writerow(info_list)
    csvFile.close()
    print("=" * 60)