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
           
#將來的榮耀
#18我想，現在的苦楚若比起將來要顯於我們的榮耀就不足介意了。
#19受造之物切望等候神的眾子顯出來。#20因為受造之物服在虛空之下，不是自己願意，乃是因那叫他如此的。 
#21但受造之物仍然指望脫離敗壞的轄制，得享神兒女自由的榮耀。#22我們知道，一切受造之物一同歎息，勞苦，直到如今。 
#23不但如此，就是我們這有聖靈初結果子的，也是自己心裏歎息，等候得着兒子的名分，乃是我們的身體得贖。 
#24我們得救是在乎盼望；只是所見的盼望不是盼望，誰還盼望他所見的呢？#25但我們若盼望那所不見的，就必忍耐等候。
#26況且，我們的軟弱有聖靈幫助；我們本不曉得當怎樣禱告，只是聖靈親自用說不出來的歎息替我們禱告。 
#27鑒察人心的，曉得聖靈的意思，因為聖靈照着神的旨意替聖徒祈求。 
#28我們曉得萬事都互相效力，叫愛神的人得益處，就是按他旨意被召的人。 
#29因為他預先所知道的人，就預先定下效法他兒子的模樣，使他兒子在許多弟兄中作長子。 
#30預先所定下的人又召他們來；所召來的人又稱他們為義；所稱為義的人又叫他們得榮耀。
            
# =============================================================================
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
if __name__ == "__main__":
    urls = generate_urls(URL, 1, 3)  #爬取1~3頁
    print(urls)
    goods = web_scraping_bot(urls)
    df = pandas.DataFrame(goods)     #用dataframe列出
    print(df)
    #for good in goods:                #用list列出
    #    print(good)
    save_to_csv(goods, "Amazon_KB_Rank.csv")   


