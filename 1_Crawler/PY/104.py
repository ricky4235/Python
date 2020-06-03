# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 11:33:20 2019

@author: 11004076
"""

import pandas as pd
import re, time, requests
from selenium import webdriver
from bs4 import BeautifulSoup

"""
設定查詢條件
"""
# 加入使用者資訊(如使用什麼瀏覽器、作業系統...等資訊)模擬真實瀏覽網頁的情況
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36'}

# 查詢的關鍵字
my_params = {'ro':'1', # 限定全職的工作，如果不限定則輸入0
             'keyword':'數據分析', # 想要查詢的關鍵字
             'area':'6001001000%2C6001002003%2C6001002007%2C6001002011%2C6001002014%2C6001002015%2C6001002021%2C6001002020%2C6001002023%2C6001002025', # 限定在台北+新北(西部)的工作
             'isnew':'1', # 只要最近一個月有更新的過的職缺
             'isnew':'10', # 只要最近一個月有更新的過的職缺
             'mode':'l'} # 清單的瀏覽模式

"""
展開所有工作清單，後續將依序開始爬蟲
"""
# 這裡會透過Selenium打開一個瀏覽器並開始跑程式~
url = requests.get('https://www.104.com.tw/jobs/search/?' , my_params, headers = headers).url
driver = webdriver.Chrome("./chromedriver")
driver.get(url)

# 網頁的設計方式是滑動到下方時，會自動加載新資料，在這裡透過程式送出Java語法幫我們執行「滑到下方」的動作
for i in range(20): 
    driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
    time.sleep(0.6)
    
# 自動加載只會加載15次，超過之後必須要點選「手動載入」的按鈕才會繼續載入新資料（可能是防止爬蟲）
k = 1
while k != 0:
    try:
        # 手動載入新資料之後會出現新的more page，舊的就無法再使用，所以要使用最後一個物件[-1]
        driver.find_elements_by_class_name("js-more-page",)[-1].click() 
        # 如果真的找不到，也可以直接找中文!
        # driver.find_element_by_xpath("//*[contains(text(),'手動載入')]").click()
        print('Click 手動載入，' + '載入第' + str(15 + k) + '頁')
        k = k+1
        time.sleep(1) # 時間設定太短的話，來不及載入新資料就會跳錯誤
    except:
        k = 0
        print('No more Job')

# 透過BeautifulSoup解析資料，並先得知有幾筆資料
soup = BeautifulSoup(driver.page_source, 'html.parser')
List = soup.findAll('a',{'class':'js-job-link'})
print('共有 ' + str(len(List)) + ' 筆資料')

"""
解析爬蟲資料並整理成DataFrame
"""
#在正式開始爬蟲之前，我預先定義一個函數，專於用來處理「職務類別」這個複選題，稍後將用這個函數將其串接在一起
# ex: 其他專案管理師、市場調查／市場分析
def bind(cate):
    k = []
    for i in cate:
        if len(i.text) > 0:
            k.append(i.text)
    return str(k)

# 開始逐筆爬資料囉!
JobList = pd.DataFrame()

i = 0
while i < len(List):
    # print('正在處理第' + str(i) + '筆，共 ' + str(len(List)) + ' 筆資料')
    content = List[i]
    # 這裡用Try的原因是，有時候爬太快會遭到系統阻擋導致失敗。因此透過這個方式，當我們遇到錯誤時，會重新再爬一次資料！
    try:
        resp = requests.get('https://' + content.attrs['href'].strip('//'))  #進入每個職缺各別頁面
        soup2 = BeautifulSoup(resp.text,'html.parser')  #剖析每個職缺各別頁面
        df = pd.DataFrame(
            data = [{
                '公司名稱':soup2.find('a', {'class':'cn'}).text,
                '工作職稱':content.attrs['title'],
                '工作內容':soup2.find('p').text,
                '職務類別':bind(soup2.findAll('dd', {'class':'cate'})[0].findAll('span')), #回到上方函數bind(cate)處理
                '工作待遇':soup2.find('dd', {'class':'salary'}).text.split('\n\n',2)[0].replace(' ',''),
                '工作性質':soup2.select('div > dl > dd')[2].text,
                '上班地點':soup2.select('div > dl > dd')[3].text.split('\n\n',2)[0].split('\n',2)[1].replace(' ',''),
                '管理責任':soup2.select('div > dl > dd')[4].text,
                '出差外派':soup2.select('div > dl > dd')[5].text,
                '上班時段':soup2.select('div > dl > dd')[6].text,
                '休假制度':soup2.select('div > dl > dd')[7].text,
                '可上班日':soup2.select('div > dl > dd')[8].text,
                '需求人數':soup2.select('div > dl > dd')[9].text,
                '接受身份':soup2.select('div.content > dl > dd')[10].text,
                '學歷要求':soup2.select('div.content > dl > dd')[12].text,
                '工作經歷':soup2.select('div.content > dl > dd')[11].text,
                '語文條件':soup2.select('div.content > dl > dd')[14].text,
                '擅長工具':soup2.select('div.content > dl > dd')[15].text,
                '工作技能':soup2.select('div.content > dl > dd')[16].text,
                '其他條件':soup2.select('div.content > dl > dd')[17].text,
                '公司福利':soup2.select('div.content > p')[1].text,
                '科系要求':soup2.select('div.content > dl > dd')[13].text,
                '聯絡方式':soup2.select('div.content')[3].text.replace('\n',''),
                '連結路徑':'https://' + content.attrs['href'].strip('//')}],
            columns = ['公司名稱','工作職稱','工作內容','職務類別','工作待遇','工作性質','上班地點','管理責任','出差外派',
                       '上班時段','休假制度','可上班日','需求人數','接受身份','學歷要求','工作經歷','語文條件','擅長工具',
                       '工作技能','其他條件','公司福利','科系要求','聯絡方式','連結路徑'])
        JobList = JobList.append(df, ignore_index=True)
        i += 1
        print("Success and Crawl Next 目前正在爬第" + str(i) + "個職缺資訊")
        time.sleep(0.5) # 執行完休息0.5秒，避免造成對方主機負擔
    except:
        print("Fail and Try Again!")
        pass   #失敗的話Try Again會一直無限迴圈，所以新增跳過再繼續
        
    

JobList

JobList.to_excel('JobList_DA2.xlsx', encoding='cp950')