from selenium import webdriver
from selenium.webdriver.common.keys import Keys

driver = webdriver.Chrome("./chromedriver")
driver.implicitly_wait(10)
url = "https://www.google.com"
driver.get(url)

keyword = driver.find_element_by_css_selector("input.gLFyf.gsfi")
keyword.send_keys("XPath")
keyword.send_keys(Keys.ENTER);

items = driver.find_elements_by_css_selector("#rso .ellip")

for item in items:
    a = item.find_elements_by_tag_name("a")   
    print(a)
    
driver.quit()