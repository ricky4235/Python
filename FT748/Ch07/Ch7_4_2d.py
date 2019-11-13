from selenium import webdriver
import os

driver = webdriver.Chrome("./chromedriver")
html_path = "file:///" +os.path.abspath("Ch7_4.html")
driver.implicitly_wait(10)
driver.get(html_path)
h3 = driver.find_element_by_tag_name("h3")
print(h3.text)
p = driver.find_element_by_tag_name("p")
print(p.text)
driver.quit()