from selenium import webdriver
import os

driver = webdriver.Chrome("./chromedriver")
html_path = "file:///" +os.path.abspath("Ch7_4.html")
driver.implicitly_wait(10)
driver.get(html_path)
user = driver.find_element_by_name("username")
print(user.tag_name)
print(user.get_attribute("type"))
eles = driver.find_elements_by_name("continue")
for ele in eles:
    print(ele.get_attribute("type"))
driver.quit()
