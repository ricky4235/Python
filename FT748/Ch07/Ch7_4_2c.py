from selenium import webdriver
import os

driver = webdriver.Chrome("./chromedriver")
html_path = "file:///" +os.path.abspath("Ch7_4.html")
driver.implicitly_wait(10)
driver.get(html_path)
link1 = driver.find_element_by_link_text('Continue')
print(link1.text)
link2 = driver.find_element_by_partial_link_text('Conti')
print(link2.text)
link3 = driver.find_element_by_link_text('取消')
print(link3.text)
link4 = driver.find_element_by_partial_link_text('取')
print(link4.text)
driver.quit()