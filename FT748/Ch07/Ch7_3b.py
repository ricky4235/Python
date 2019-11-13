from selenium import webdriver
from bs4 import BeautifulSoup

driver = webdriver.Chrome("./chromedriver")
driver.implicitly_wait(10)
driver.get("http://example.com")
print(driver.title)
soup = BeautifulSoup(driver.page_source, "lxml")
fp = open("index.html", "w", encoding="utf8")
fp.write(soup.prettify())
print("寫入檔案index.html...")
fp.close()
driver.quit()