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
    all_goods = [["rank, name, star, reviews, price"]]
    page = 1
    
    for url in urls:

        print("sd: d" + str(page) + "html.")
        page = page + 1
        r = get_resource(url)
        if r.status_code == requests.codes.ok:
            
            soup = parse_html(r.text)
            goods = get_goods(soup)
            all_goods = all_goods + goods
            print("wait 8 second...")
            if soup.find("li", class_="a-disabled a-last"):
                break
            time.sleep(5)
        else:
        
            print("HTTP request error...")

    return all_goods

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
    
# 1太初有道，道與神同在，道就是神。 2這道太初與神同在。 
# 3萬物是藉着他造的；凡被造的，沒有一樣不是藉着他造的。 
# 4生命在他裏頭，這生命就是人的光。 5光照在黑暗裏，黑暗卻不接受光。
# 6有一個人，是從神那裏差來的，名叫約翰。 
# 7這人來，為要作見證，就是為光作見證，叫眾人因他可以信。 
# 8他不是那光，乃是要為光作見證。 9那光是真光，照亮一切生在世上的人。 
# 10他在世界，世界也是藉着他造的，世界卻不認識他。 
# 11他到自己的地方來，自己的人倒不接待他。 
# 12凡接待他的，就是信他名的人，他就賜他們權柄作神的兒女。 
# 13這等人不是從血氣生的，不是從情慾生的，也不是從人意生的，乃是從神生的。
# 14道成了肉身，住在我們中間，充充滿滿地有恩典有真理。我們也見過他的榮光，正是父獨生子的榮光。
