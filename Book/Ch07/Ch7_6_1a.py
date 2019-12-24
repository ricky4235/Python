from selenium import webdriver

driver = webdriver.Chrome("./chromedriver")
driver.implicitly_wait(10)
url = "https://hahow.in/courses"
driver.get(url)

items = driver.find_elements_by_css_selector("h4.title")
                                            
for item in items:
    print(item.text)                    

driver.quit()