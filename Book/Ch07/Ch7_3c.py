from selenium import webdriver

driver = webdriver.Chrome("./chromedriver")
driver.implicitly_wait(10)
driver.get("http://example.com")
# 使用Selenium的定位函數
h1 = driver.find_element_by_tag_name("h1")
print(h1.text)
p = driver.find_element_by_tag_name("p")
print(p.text)
driver.quit()