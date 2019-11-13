# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 15:41:09 2019

@author: 11004076
"""

# import libs
import requests
import pandas as pd
from bs4 import BeautifulSoup
# define url for crawling
url = 'https://movies.yahoo.com.tw/chart.html'

# GET request from url and parse via BeautifulSoup
resp = requests.get(url)
resp.encoding = 'utf-8' # encoded with format utf-8 for chinese character
soup = BeautifulSoup(resp.text, 'lxml')

# parse colname 
rows = soup.find_all('div', class_='tr')
# get strings and convert into list
colname = list(rows.pop(0).stripped_strings) #把第一row pop出來當作是column name:['本週', '上週', '片名', '上映日期', '預告片', '網友滿意度']
print(colname)
#觀察html發現，是第一名的電影的電影名稱被包在更裡面的區塊中，而其他資料大致上的位置都差不多。
#由於這個網頁上很多資料的排列是靠位置來組成的，並沒有更進一步的定義每一個資料的class attribute，而都是class = “td”，
#所以這種時候透過find_next依序的來爬對應的資料就會比較好用，若有定義明確的屬性值的話則可以寫進attrs中即可直接爬到對應的資料，這邊則是需要一個一個find_next往下找。

contents = []
for row in rows:
    thisweek_rank = row.find_next('div', attrs={'class':'td'})
    updown = thisweek_rank.find_next('div')
    lastweek_rank = updown.find_next('div')
    
    #for the data form of first row in this web page is different from other rows
    if thisweek_rank.string == str(1):
        movie_title = lastweek_rank.find_next('h2') #or dd
    else:
        movie_title = lastweek_rank.find_next('div', attrs={'class':'rank_txt'})
        
    release_data = movie_title.find_next('div', attrs={'class':'td'})
    trailer = release_data.find_next('div', attrs={'class':'td'})
    trailer_address = trailer.find_next('a')['href']
    stars = row.find('h6', attrs={'class':'count'})
    
    #replace None with empty string ''
    lastweek_rank = lastweek_rank.string if lastweek_rank.string else ''
    
    c = [thisweek_rank.string, lastweek_rank, movie_title.string, release_data.string, trailer_address, stars.string]
    print(c)
    contents.append(c)
    
#因為在parse的時候已經將content的格式存成list，所以接下來只要將contents丟給pandas的DataFrame即可無痛轉換成DataFrame的形式，別忘記要指定columns的名稱。
#convert to data frame format
df = pd.DataFrame(contents, columns=colname)
df.head()

#最後再將df output成csv格式就完成了!
import os
import datetime

cwd = os.getcwd()
timestamp = datetime.datetime.now()
timestamp = timestamp.strftime('%Y%m%d')

filename = os.path.join(cwd, 'yahoo_movie_rank_{}.csv'.format(timestamp))
df.to_csv(filename,encoding='utf_8_sig', index=False)
print('Save csv to{}'.format(filename))