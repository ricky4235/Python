from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
import time

driver = webdriver.Chrome("./chromedriver")
driver.implicitly_wait(10)
url = "https://www.python.org/"
driver.get(url)

menu = driver.find_element_by_css_selector("#about")
item = driver.find_element_by_css_selector("#about>ul>li.tier-2.element-1")

actions = ActionChains(driver)
actions.move_to_element(menu)
actions.click(item)
actions.perform()
time.sleep(5)
driver.quit()