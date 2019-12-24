from selenium import webdriver

driver = webdriver.Chrome("./chromedriver")
driver.implicitly_wait(10)
url = "https://github.com/login"
driver.get(url)

username = "hueyan@ms2.hinet.net"
password = "********"
user = driver.find_element_by_css_selector("#login_field")
user.send_keys(username)
pwd = driver.find_element_by_css_selector("#password")
pwd.send_keys(password)
button = driver.find_element_by_css_selector("input.btn.btn-primary.btn-block")
button.click()

items = driver.find_elements_by_xpath("//header/div/div[2]/div[1]/ul/li/a")

for item in items:
    print(item.text)
    print(item.get_attribute("href"))
    
driver.quit()