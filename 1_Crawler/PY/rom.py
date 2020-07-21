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

    
# =============================================================================
# 
# 羅馬書 8
# 在聖靈裏的生活
# 1如今，那些在基督耶穌裏的人就不被定罪了。 2因為賜生命的聖靈的律，在基督耶穌裏從罪和死的律中把你釋放出來。 
#3律法既因肉體軟弱而無能為力，　神就差遣自己的兒子成為罪身的樣子，為了對付罪，在肉體中定了罪， 
#4為要使律法要求的義，實現在我們這不隨從肉體、只隨從聖靈去行的人身上。 
#5因為，隨從肉體的人體貼肉體的事；隨從聖靈的人體貼聖靈的事。 6體貼肉體就是死；體貼聖靈就是生命和平安
#7因為體貼肉體就是與　神為敵，對　神的律法不順服，事實上也無法順服。 8屬肉體的人無法使　神喜悅。
# 9如果　神的靈住在你們裏面，你們就不屬肉體，而是屬聖靈了。人若沒有基督的靈，就不是屬基督的。 
#10基督若在你們裏面，身體就因罪而死，靈卻因義而活。 11然而，使耶穌從死人中復活的　神的靈若住在你們裏面，那使基督從死人中復活的，也必藉着住在你們裏面的聖靈使你們必死的身體又活過來。
# 12弟兄們，這樣看來，我們不是欠肉體的債去順從肉體而活。 13你們若順從肉體活着，必定會死；若靠着聖靈把身體的惡行處死，就必存活。 14因為凡被　神的靈引導的都是　神的兒子。 15你們所領受的不是奴僕的靈，仍舊害怕；所領受的是兒子名分的靈，因此我們呼叫：「阿爸，父！」 16聖靈自己與我們的靈一同見證我們是　神的兒女。 17若是兒女，就是後嗣，是　神的後嗣，和基督同作後嗣。如果我們和他一同受苦，是要我們和他一同得榮耀。
# 將來的榮耀
# 18我認為，現在的苦楚，若比起將來要顯示給我們的榮耀，是不足介意的。 19受造之物切望等候　神的眾子顯出來。 20-21因為受造之物屈服在虛空之下，不是自己願意，而是因那使它屈服的叫他如此。但受造之物仍然指望從敗壞的轄制下得釋放，得享　神兒女榮耀的自由。 22我們知道，一切受造之物一同呻吟，一同忍受陣痛，直到如今。 23不但如此，就是我們這有聖靈作初熟果子的，也是自己內心呻吟，等候得着兒子的名分，就是我們的身體得救贖。 24我們得救是在於盼望；可是看得見的盼望就不是盼望。誰還去盼望他所看得見的呢？ 25但我們若盼望那看不見的，我們就耐心等候。
# 26同樣，我們的軟弱有聖靈幫助。我們本不知道當怎樣禱告，但是聖靈親自用無可言喻的嘆息替我們祈求。 27那鑒察人心的知道聖靈所體貼的，因為聖靈照着　神的旨意替聖徒祈求。 28我們知道，萬事8．28「萬事」：有古卷是「　神使萬事」。都互相效力，叫愛　神的人得益處，就是按他旨意被召的人。 29因為他所預知的人，他也預定他們效法他兒子的榜樣，使他兒子在許多弟兄中作長子8．29「作長子」或譯「為首生者」。。 30他所預定的人，他又召他們來；所召來的人，他又稱他們為義；所稱為義的人，他又叫他們得榮耀。
# 不能隔絕的愛
# 31既是這樣，我們對這些事還要怎麼說呢？　神若幫助我們，誰能抵擋我們呢？ 32 　神既不顧惜自己的兒子，為我們眾人捨了他，豈不也把萬物和他一同白白地賜給我們嗎？ 33誰能控告　神所揀選的人呢？有　神稱他們為義了。 34誰能定他們的罪呢？有基督耶穌8．34有古卷沒有「耶穌」。已經死了，而且復活了，現今在　神的右邊，也替我們祈求。 35誰能使我們與基督的愛隔絕呢？難道是患難嗎？是困苦嗎？是迫害嗎？是飢餓嗎？是赤身露體嗎？是危險嗎？是刀劍嗎？ 36如經上所記：
# 「我們為你的緣故終日被殺；
# 人看我們如將宰的羊。」
# 37然而，靠着愛我們的主，在這一切的事上，我們已經得勝有餘了。 38因為我深信，無論是死，是活，是天使，是掌權的，是有權能的8．38「是掌權的，是有權能的」：指靈界的勢力。，是現在的事，是將來的事， 39是高處的，是深處的，是別的受造之物，都不能使我們與　神的愛隔絕，這愛是在我們的主基督耶穌裏的。
# =============================================================================
