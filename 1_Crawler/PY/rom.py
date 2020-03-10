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
    all_goods = [["rank, name, star, reviews, price]]  #巢狀清單
    page = 1
    
    for url in urls:
        print("sd: d" + str(page) + "html.")
        page = page + 1
        r = get_resource(url)
        if r.status_code == requests.codes.ok:
            soup = parse_html(r.text)
            goods = get_goods(soup)
            all_goods = all_goods + goods
            print("等待5秒鐘...")
            if soup.find("li", class_="a-disabled a-last"):
                break   #已經沒有下一頁
            time.sleep(5) 
        else:
            print("HTTP請求錯誤...")

    return all_goods

def save_to_csv(items, file):
    with open(file, "w+", newline="", encoding="utf_8_sig") as fp:  #utf_8_sig:能讓輸出的csv正確顯示中文(utf_8會有亂碼)
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



#在聖靈裏的生活
#律#1如今，那些在基督耶穌裏的就不定罪了。 2因為賜生命聖靈的律，在基督耶穌裏釋放了我，使我脫離罪和死的"律"了。 
#肉#3"律法"既因肉體軟弱，有所不能行的，神就差遣自己的兒子，成為罪身的形狀，作了贖罪祭，在肉體中定了罪案， 
#隨#4"使律法的義"成就在我們這不"隨從"肉體、只隨從聖靈的人身上。 5因為，隨從肉體的人體貼肉體的事；隨從聖靈的人"體貼"聖靈的事。 
#體#6"體貼"肉體的，就是死；體貼聖靈的，乃是生命、平安。 7原來體貼肉體的，就是與神為仇；因為不服神的律法，也是不能服， 
#屬#8而且"屬"肉體的人不能得神的喜歡。 9如果"神的靈"住在你們心裏，你們就不屬肉體，乃屬聖靈了。
#基# 人若沒有"基督的靈"，就不是屬基督的。10基督若在你們心裏，身體就因罪而死，心靈卻因義而活。 
#耶#11然而，叫"耶穌從死裏復活者的靈"若住在你們心裏，那叫基督耶穌從死裏復活的，也必藉着住在你們心裏的聖靈，使你們必死的身體又活過來。
#順#12弟兄們，這樣看來，我們並不是欠肉體的債去順從肉體活着。 13你們若順從肉體活着，必要死；若靠着聖靈治死身體的惡行，必要活着。 
#子#14因為凡被神的靈引導的，都是神的兒子。 15你們所受的，不是奴僕的心，仍舊害怕；所受的，乃是兒子的心，因此我們呼叫：「阿爸！父！」 
#女#16聖靈與我們的心同證我們是神的兒女； 17既是兒女，便是後嗣，就是神的後嗣，和基督同作後嗣。如果我們和他一同受苦，也必和他一同得榮耀。