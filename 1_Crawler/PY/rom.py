# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 17:02:41 2020

@author: 11004076
"""

def get_resource(url):
    headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
               "AppleWebKit/537.36 (KHTML, like Gecko)"
               "Chrome/63.0.3239.132 Safari/537.36"}
    return requests.get(url, headers=headers) 

def parse_html(html_str):
    return BeautifulSoup(html_str, "lxml")

def get_goods(soup):
    goods = []
    rows = soup.find_all("li", class_="zg-item-immersion")
    for row in rows:
        rank = row.find("span", class_="zg-badge-text").text
        name = row.find("div", class_="a-section a-spacing-small").img["alt"]
        star = row.find("span", class_="a-icon-alt").text
        reviews = row.find("a", class_="a-size-small a-link-normal").text
        price = row.find("span", class_="p13n-sc-price").text
        
        good= [rank, name, star, reviews, price]
        goods.append(good)
    return goods

def web_scraping_bot(urls):
    all_goods = [["rank, name, star, reviews, price]] 
    page = 1
    
    for url in urls:
        print("sd: d" + str(page) + "html.")
        page = page + 1
        r = get_resource(url)
        if r.status_code == requests.codes.ok:
            soup = parse_html(r.text)
            goods = get_goods(soup)
            all_goods = all_goods + goods
            print("wait 5 second...")
            if soup.find("li", class_="a-disabled a-last"):
                break
            time.sleep(5)
        else:
            print("HTTP request error...")


    return all_goods
def

def save_to_csv(items, file):
    with open(file, "w+", newline="", encoding="utf_8_sig") as fp:
        writer = csv.writer(fp)
        for item in items:
            writer.writerow(item)

if __name__ == "__main__":
    urls = generate_urls(URL, 1, 3)  #爬取1~3頁
    print(urls)
    goods = web_scraping_bot(urls)
    df = pandas.DataFrame(goods)     #用dataframe列出
    print(df)
    #for good in goods:                #用list列出
    #    print(good)
    save_to_csv(goods, "Amazon_KB_Rank.csv") 
    
#神的愛
#31既是這樣，還有甚麼說的呢？神若幫助我們，誰能敵擋我們呢？ 
#32神既不愛惜自己的兒子，為我們眾人捨了，豈不也把萬物和他一同白白地賜給我們嗎？ 
#33誰能控告神所揀選的人呢？有神稱他們為義了
#34誰能定他們的罪呢？有基督耶穌已經死了，而且從死裏復活，現今在神的右邊，也替我們祈求
#35誰能使我們與基督的愛隔絕呢？難道是患難嗎？是困苦嗎？是逼迫嗎？是飢餓嗎？是赤身露體嗎？是危險嗎？是刀劍嗎？ 
#36如經上所記：我們為你的緣故終日被殺；人看我們如將宰的羊。
#37然而，靠着愛我們的主，在這一切的事上已經得勝有餘了。 
#38因為我深信無論是死，是生，是天使，是掌權的，是有能的，是現在的事，是將來的事， 
#39是高處的，是低處的，是別的受造之物，都不能叫我們與神的愛隔絕；這愛是在我們的主基督耶穌裏的。