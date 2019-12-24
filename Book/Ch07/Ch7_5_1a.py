from selenium import webdriver

driver = webdriver.Chrome("./chromedriver")
driver.implicitly_wait(10)
url = "https://www.google.com"
driver.get(url)

keyword = driver.find_element_by_css_selector("#lst-ib")
keyword.send_keys("XPath")
button = driver.find_element_by_css_selector("input[type='submit']")
button.click()

items = driver.find_elements_by_xpath("//div[@class='r']")

for item in items:
    h3 = item.find_element_by_tag_name("h3")
    print(h3.text)
    a = item.find_element_by_tag_name("a")   
    print(a.get_attribute("href"))
    
driver.quit()