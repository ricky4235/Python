from selenium import webdriver
from bs4 import BeautifulSoup

driver = webdriver.Chrome("./chromedriver")
driver.implicitly_wait(10)
driver.get("http://example.com")
# 使用Beautiful Soup剖析HTML網頁
soup = BeautifulSoup(driver.page_source, "lxml")
tag_h1 = soup.find("h1") 
print(tag_h1.string)
tag_p = soup.find("p") 
print(tag_p.string)
driver.quit()