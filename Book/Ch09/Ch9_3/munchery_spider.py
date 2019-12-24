import time
import csv
from selenium import webdriver
# 目標URL網址
URL = "https://munchery.com/"

class DishesSpider():
    def __init__(self, url):
        self.url_to_crawl = url
        self.all_items = [["名稱","網址","圖片"]]

    def start_driver(self):
        print("啟動 WebDriver...")
        self.driver = webdriver.Chrome("./chromedriver")
        self.driver.implicitly_wait(10)

    def close_driver(self):
        self.driver.quit()
        print("關閉 WebDriver...")
        
    def get_page(self, url):
        print("取得網頁...")
        self.driver.get(url)
        time.sleep(2)

    def login(self):
        print("登入網站...")
        try:
            form = self.driver.find_element_by_xpath('//*[@class="signup-login-form"]')
            email = form.find_element_by_xpath('.//*[@class="user-input email"]')
            email.send_keys('hueyan@ms2.hinet.net')
            zipcode = form.find_element_by_xpath('.//*[@class="user-input zip-code"]')
            zipcode.send_keys('12345')
            button = form.find_element_by_xpath('.//button[@class="large orange button view-menu-button"]')
            button.click()
            print("成功登入網站...")
            time.sleep(5)            
            return True
        except Exception:
            print("登入網站失敗...")
            return False

    def grab_dishes(self):
        print("開始爬取食譜項目...")
        for div in self.driver.find_elements_by_xpath('//a[@class="menu-item"]'):
            item = self.process_item(div)
            if item:
                self.all_items.append(item)

    def process_item(self, div):
        item = []
        try:
            url = div.get_attribute("href")
            image = div.find_element_by_xpath('.//img[@class="item-photo"]').get_attribute("src")
            title = div.find_element_by_xpath('.//div[@class="item-title"]').text            
            item = [title, image, url]
            
            return item            
        except Exception:
            return False

    def parse_dishes(self):
        self.start_driver()     # 開啟 WebDriver
        self.get_page(self.url_to_crawl)
        if self.login():        # 是否成功登入
            self.grab_dishes()  # 爬取食譜
        self.close_driver()     # 關閉 WebDriver
        if self.all_items:
            return self.all_items
        else:
            return []
        
def save_to_csv(items, file):
    with open(file, "w+", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        for item in items:
            writer.writerow(item)
            
if __name__ == '__main__':            
    Munchery = DishesSpider(URL)
    dishes = Munchery.parse_dishes()
    for item in dishes:
        print(item)
    save_to_csv(dishes, "dishes.csv")    