from selenium import webdriver
import os

driver = webdriver.Chrome("./chromedriver")
html_path = "file:///" +os.path.abspath("Ch7_4.html")
driver.implicitly_wait(10)
driver.get(html_path)
content = driver.find_element_by_css_selector("h3.content")
print(content.text)
p = driver.find_element_by_css_selector("p")
print(p.text)
driver.quit()