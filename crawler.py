#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 19:25:43 2019

@author: ricky
"""

"""
抓取PTT電影版的網頁原始碼(HTML)
"""
import urllib.request as req
url="https://www.ptt.cc/bbs/movie/index.html"
#建立一個 Request 物件(為了讓網站判斷我們是正常使用者)
#附加 Request Headers 的資訊(到網站的開發人員找User-Agent後的東西複製過來，要讓網站知道我們電腦的作業系統和使用瀏覽器)
request=req.Request(url, headers={
        "User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36"
})
with req.urlopen(request) as response:
    data=response.read().decode("utf-8")
print(data)

"""
解析原始碼，取得每篇文章的標題
"""
import bs4
root=bs4.BeautifulSoup(data, "html.parser")  #讓BeautifulSoup協助我們解析HTML格式文件
titles=root.find_all("div", class_="title")  #尋找所有class="title"的div標籤
for title in titles:
    if title.a != None: #如果標題包含a標籤(沒有被刪除)，就印出來
        print(title.a.string)
