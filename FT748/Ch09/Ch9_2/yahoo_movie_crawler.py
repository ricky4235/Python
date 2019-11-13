import time
import requests
import csv
import re
from bs4 import BeautifulSoup

# 目標URL網址
URL = "https://movies.yahoo.com.tw/movie_thisweek.html?page={0}"

def generate_urls(url, start_page, end_page):
    urls = []
    for page in range(start_page, end_page+1):
        urls.append(url.format(page))
    return urls

def get_resource(url):
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
               "AppleWebKit/537.36 (KHTML, like Gecko)"
               "Chrome/63.0.3239.132 Safari/537.36"}
    return requests.get(url, headers=headers) 

def parse_html(html_str):
    return BeautifulSoup(html_str, "lxml")

def format_date(date_str):
    # 取出上映日期
    pattern = '\d+-\d+-\d+'
    match = re.search(pattern, date_str)
    if match is None:
        return date_str
    else:
        return match.group(0)

def get_movies(soup):
    movies = []
    rows = soup.find_all("div", class_="release_info_text")
    for row in rows:
        movie_name_div = row.find("div", class_="release_movie_name")
        cht_name = movie_name_div.a.text.strip()
        eng_name = movie_name_div.find("div", class_="en").a.text.strip()
        expectation = row.find("div", class_="leveltext").span.text.strip()
        photo = row.parent.find_previous_sibling(
                "div", class_="release_foto")
        poster_url = photo.a.img["src"]
        release_date = format_date(row.find('div', 'release_movie_time').text)
        
        movie= [cht_name,eng_name,expectation,
                poster_url,release_date]
        movies.append(movie)
    return movies

def save_to_csv(items, file):
    with open(file, "w+", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        for item in items:
            writer.writerow(item)

def web_scraping_bot(urls):
    all_movies = [["中文片名","英文片名","期待度","海報圖片","上映日"]]
    page = 1
    
    for url in urls:
        print("抓取: 第" + str(page) + "頁 網路資料中...")
        page = page + 1
        r = get_resource(url)
        if r.status_code == requests.codes.ok:
            soup = parse_html(r.text)
            movies = get_movies(soup)
            all_movies = all_movies + movies
            print("等待5秒鐘...")
            if soup.find("li", class_="nexttxt disabled"):
                break   # 已經沒有下一頁
            time.sleep(5) 
        else:
            print("HTTP請求錯誤...")

    return all_movies

if __name__ == "__main__":
    urls = generate_urls(URL, 1, 5)
    # print(urls)
    movies = web_scraping_bot(urls)
    for movie in movies:
        print(movie)
    save_to_csv(movies, "movies.csv")
