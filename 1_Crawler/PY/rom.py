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
        
#1奉神旨意，作基督耶穌使徒的保羅，寫信給在以弗所的聖徒，就是在基督耶穌裏有忠心的人。 
#2願恩惠、平安從神我們的父和主耶穌基督歸與你們！
        #基督裏的屬靈福氣
#3願頌讚歸與我們主耶穌基督的父神！他在基督裏曾賜給我們天上各樣屬靈的福氣： 
#4就如神從創立世界以前，在基督裏揀選了我們，使我們在他面前成為聖潔，無有瑕疵； 
#5又因愛我們，就按着自己的意旨所喜悅的，預定我們藉着耶穌基督得兒子的名分，
        
#6使他榮耀的恩典得着稱讚；這恩典是他在愛子裏所賜給我們的。 
#7我們藉這愛子的血得蒙救贖，過犯得以赦免，乃是照他豐富的恩典。 
#8這恩典是神用諸般智慧聰明，充充足足賞給我們的； 
#9都是照他自己所預定的美意，叫我們知道他旨意的奧祕， 
#10要照所安排的，在日期滿足的時候，使天上、地上、一切所有的都在基督裏面同歸於一。 
#11我們也在他裏面得得：或譯成了基業；這原是那位隨己意行、做萬事的，照着他旨意所預定的， 
#12叫他的榮耀從我們這首先在基督裏有盼望的人可以得着稱讚。 
#13你們既聽見真理的道，就是那叫你們得救的福音，也信了基督，既然信他，就受了所應許的聖靈為印記。
#14這聖靈是我們得基業的憑據原文是質，直等到神之民民：原文是產業被贖，使他的榮耀得着稱讚。
 
 #仁愛、喜樂、和平、忍耐、恩慈、良善、信實、溫柔、節制