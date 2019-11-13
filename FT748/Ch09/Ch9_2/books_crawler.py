import time
import requests
import csv
from bs4 import BeautifulSoup

# 目標URL網址
URL = "http://search.books.com.tw/search/query/key/{0}/cat/all"

def generate_search_url(url, keyword):
    url = url.format(keyword)
    
    return url

def get_resource(url):
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
               "AppleWebKit/537.36 (KHTML, like Gecko)"
               "Chrome/63.0.3239.132 Safari/537.36"}
    return requests.get(url, headers=headers)

def parse_html(r):
    if r.status_code == requests.codes.ok:
        r.encoding = "utf8"
        soup = BeautifulSoup(r.text, "lxml")        
    else:
        print("HTTP請求錯誤..." + url)
        soup = None
    
    return soup    

def get_ISBN(url):
    soup = parse_html(get_resource("http:" + url))
    if soup != None:
        try:
           isbn = soup.find(itemprop="productID")["content"][5:]
        except:
           isbn = "0000"
    else:
        isbn = "1111"
    return isbn

def save_to_csv(items, file):
    with open(file, 'w+', newline='') as fp:
        writer = csv.writer(fp)
        for item in items:
            writer.writerow(item)
            
def web_scraping_bot(url):
    booklist = [["書名","ISBN","網址","書價"]]
    print("抓取網路資料中...")
    soup = parse_html(get_resource(url))
    if soup != None:
        # print(soup)
        tag_item = soup.find_all(class_="item")
        for item in tag_item:
            book = []
            book.append(item.find("img")["alt"])
            book.append(get_ISBN(item.find("a")["href"]))
            book.append("http:" + item.find("a")["href"])
            price = item.find(class_="price").find_all("b")
            if len(price) == 1:
                book.append(price[0].string)
            else:
                book.append(price[1].string)
            booklist.append(book)
            print("等待2秒鐘...")
            time.sleep(2) 

    return booklist

if __name__ == '__main__':
    url = generate_search_url(URL, "演算法")
    print(url)
    booklist = web_scraping_bot(url)
    for item in booklist:
        print(item)
    save_to_csv(booklist, "booklist.csv")
