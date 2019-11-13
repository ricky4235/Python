import time
import requests
import csv
from bs4 import BeautifulSoup

# 目標URL網址
URL = "https://tw.stock.yahoo.com/q/q?s="

def generate_urls(url, stocks):
    urls = []
    for stock in stocks:
        urls.append(url + stock)
    return urls

def get_resource(url):
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
               "AppleWebKit/537.36 (KHTML, like Gecko)"
               "Chrome/63.0.3239.132 Safari/537.36"}
    return requests.get(url, headers=headers) 

def parse_html(html_str):
    return BeautifulSoup(html_str, "lxml")

def get_stock(soup, stock_id):
    table = soup.find_all(text="成交")[0].parent.parent.parent
    status = table.select("tr")[0].select("th")[2].text
    name =   table.select("tr")[1].select("td")[0].text
    price =  table.select("tr")[1].select("td")[2].text
    yclose = table.select("tr")[1].select("td")[7].text
    volume = table.select("tr")[1].select("td")[6].text
    high =   table.select("tr")[1].select("td")[9].text
    low  =   table.select("tr")[1].select("td")[10].text
    
    return [stock_id, name[4:-6], status, price, yclose, volume, high, low]

def save_to_csv(items, file):
    with open(file, "w+", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        for item in items:
            writer.writerow(item)

def web_scraping_bot(urls):
    stocks = [["代碼","名稱","狀態","股價","昨收","張數","最高","最低"]]
    
    for url in urls:
        stock_id = url.split("=")[-1]
        print("抓取: " + stock_id + " 網路資料中...")
        r = get_resource(url)
        if r.status_code == requests.codes.ok:
            soup = parse_html(r.text)
            stock = get_stock(soup, stock_id)
            stocks.append(stock)
            print("等待5秒鐘...")
            time.sleep(5) 
        else:
            print("HTTP請求錯誤...")       

    return stocks

if __name__ == "__main__":
    urls = generate_urls(URL, ["3711", "2330", "2454"])
    # print(urls)
    stocks = web_scraping_bot(urls)
    for stock in stocks:
        print(stock)
    save_to_csv(stocks, "stocks.csv")
