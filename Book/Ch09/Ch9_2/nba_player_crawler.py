import time
import requests
import csv
from bs4 import BeautifulSoup

# 目標URL網址
URL = "https://www.basketball-reference.com/teams/{0}/2018.html"
TEAMS = ["CLE", "HOU", "GSW"]

def generate_urls(url, teams):
    urls = []
    for team in teams:
        urls.append(url.format(team))
    return urls

def get_resource(url):
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
               "AppleWebKit/537.36 (KHTML, like Gecko)"
               "Chrome/63.0.3239.132 Safari/537.36"}
    return requests.get(url, headers=headers) 

def parse_html(html_str):
    return BeautifulSoup(html_str, "lxml")

def get_players(soup, team):
    team_players = []
    table = soup.find(id="roster")  # 找到表格
    # HTML表格的所有列
    for row in table.find("tbody").find_all("tr"): 
        no = row.th.text  # 背號
        cols = row.findAll("td")
        # 球隊, 背號, 姓名, 位置, 體重, 生日, 經驗, 大學
        team_players.append([team, no, cols[0].text, cols[1].text, 
                             cols[3].text, cols[4].text, 
                             cols[6].text, cols[7].text])
            
    return team_players

def save_to_csv(items, file):
    with open(file, "w+", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        for item in items:
            writer.writerow(item)

def web_scraping_bot(urls):
    count = 0    
    total_players=[["球隊","背號","姓名","位置","體重","生日","經驗","大學"]]
    
    for url in urls:
        team_name = TEAMS[count]
        count = count + 1;
        print("抓取: " + team_name + " 網路資料中...")
        r = get_resource(url)
        if r.status_code == requests.codes.ok:
            soup = parse_html(r.text)
            players = get_players(soup, team_name)
            total_players = total_players + players
            print("等待5秒鐘...")
            time.sleep(5)            
        else:
            print("HTTP請求錯誤...")       

    return total_players

if __name__ == "__main__":
    urls = generate_urls(URL, TEAMS)
    # print(urls)
    players = web_scraping_bot(urls)
    for item in players:
        print(item)
    save_to_csv(players, "players.csv")
