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
#
#將來的榮耀
#18我想，現在的苦楚若比起將來要顯於我們的榮耀就不足介意了。
#19受造之物切望等候神的眾子顯出來。 #20因為受造之物服在虛空之下，不是自己願意，乃是因那叫他如此的。 
#21但受造之物仍然指望脫離敗壞的轄制，得享神兒女自由的榮耀。#22我們知道，一切受造之物一同歎息，勞苦，直到如今。 
#23不但如此，就是我們這有聖靈初結果子的，也是自己心裏歎息，等候得着兒子的名分，乃是我們的身體得贖。 
#24我們得救是在乎盼望；只是所見的盼望不是盼望，誰還盼望他所見的呢？#25但我們若盼望那所不見的，就必忍耐等候。
#26況且，我們的軟弱有聖靈幫助；我們本不曉得當怎樣禱告，只是聖靈親自用說不出來的歎息替我們禱告。 
#27鑒察人心的，曉得聖靈的意思，因為聖靈照着神的旨意替聖徒祈求。 
#28我們曉得萬事都互相效力，叫愛神的人得益處，就是按他旨意被召的人。 
#29因為他預先所知道的人，就預先定下效法他兒子的模樣，使他兒子在許多弟兄中作長子。 
#30預先所定下的人又召他們來；所召來的人又稱他們為義；所稱為義的人又叫他們得榮耀。