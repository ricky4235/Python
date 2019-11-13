# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 18:01:35 2019

@author: 11004076
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 12:05:02 2019

@author: 11004076
"""
import requests
from bs4 import BeautifulSoup

# 下載 Yahoo 首頁內容
r = requests.get('https://tw.yahoo.com/')

# 確認是否下載成功
if r.status_code == requests.codes.ok:
  # 以 BeautifulSoup 解析 HTML 程式碼
  soup = BeautifulSoup(r.text, 'html.parser')

  # 以 CSS 的 class 抓出各類頭條新聞
  stories = soup.find('a', class_='story-title')
    print("標題：" ,stories.text)
