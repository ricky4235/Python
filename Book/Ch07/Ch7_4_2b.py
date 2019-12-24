from selenium import webdriver
import os

driver = webdriver.Chrome("./chromedriver")
html_path = "file:///" +os.path.abspath("Ch7_4.html")
driver.implicitly_wait(10)
driver.get(html_path)
# 定位<form>標籤
form1 = driver.find_element_by_xpath("/html/body/form[1]")
print(form1.tag_name)
form2 = driver.find_element_by_xpath("//form[1]")
print(form2.tag_name)
form3 = driver.find_element_by_xpath("//form[@id='loginForm']")
print(form3.tag_name)
# 定位密碼欄位
pwd1 = driver.find_element_by_xpath("//form/input[2][@name='password']")
print(pwd1.get_attribute("type"))
pwd2 = driver.find_element_by_xpath("//form[@id='loginForm']/input[2]")
print(pwd2.get_attribute("type"))
pwd3 = driver.find_element_by_xpath("//input[@name='password']")
print(pwd3.get_attribute("type"))
# 定位清除按鈕
clear1 = driver.find_element_by_xpath("//input[@name='continue'][@type='button']")
print(clear1.get_attribute("type"))
clear2 = driver.find_element_by_xpath("//form[@id='loginForm']/input[4]")
print(clear2.get_attribute("type"))
driver.quit()
