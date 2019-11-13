from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
import os

driver = webdriver.Chrome("./chromedriver")
html_path = "file:///" +os.path.abspath("Ch7_4.html")
driver.implicitly_wait(10)
driver.get(html_path)
try:
    content = driver.find_element_by_css_selector("h2.content")
    print(content.text)
except NoSuchElementException:
    print("選取的元素不存在...")
driver.quit()
