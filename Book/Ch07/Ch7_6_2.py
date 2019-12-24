from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time

driver = webdriver.Chrome("./chromedriver")
driver.implicitly_wait(10)
driver.get("http://stats.nba.com/players/traditional/?sort=PTS&dir=-1")

pages_remaining = True
page_num = 1
while pages_remaining:
    # 使用Beautiful Soup剖析HTML網頁
    soup = BeautifulSoup(driver.page_source, "lxml")
    table = soup.select_one("body > main > div.stats-container__inner > div > div.row > div > div > nba-stat-table > div.nba-stat-table > div.nba-stat-table__overflow > table") 
    df = pd.read_html(str(table))
    # print(df[0].to_csv())
    df[0].to_csv("ALL_players_stats" + str(page_num) + ".csv")
    print("儲存頁面:", page_num)
    try:
        # 自動按下一頁按鈕
        next_link = driver.find_element_by_xpath('/html/body/main/div[2]/div/div[2]/div/div/nba-stat-table/div[3]/div/div/a[2]')
        next_link.click()
        time.sleep(5)
        if page_num < 11:
            page_num = page_num + 1
        else:
            pages_remaining = False
    except Exception:
        pages_remaining = False  
        
driver.close()