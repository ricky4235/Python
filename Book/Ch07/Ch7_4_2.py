from selenium import webdriver
import os

driver = webdriver.Chrome("./chromedriver")
html_path = "file:///" +os.path.abspath("Ch7_4.html")
driver.implicitly_wait(10)
driver.get(html_path)
form = driver.find_element_by_id("loginForm")
print(form.tag_name)
print(form.get_attribute("id"))
driver.quit()
