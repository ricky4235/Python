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
    # 在基督裏合一
# 11所以你們應當記念：你們從前按肉體是外邦人，是稱為沒受割禮的；
    #這名原是那些憑人手在肉身上稱為受割禮之人所起的。 
#12那時，你們與基督無關，在以色列國民以外，在所應許的諸約上是局外人，並且活在世上沒有指望，沒有神。 
#13你們從前遠離神的人，如今卻在基督耶穌裏，靠着他的血，已經得親近了。
    
#14因他使我們和睦，將兩下合而為一，拆毀了中間隔斷的牆； 
#15而且以自己的身體廢掉冤仇，就是那記在律法上的規條，為要將兩下藉着自己造成一個新人，如此便成就了和睦。
#16既在十字架上滅了冤仇，便藉這十字架使兩下歸為一體，與神和好了， 
#17並且來傳和平的福音給你們遠處的人，也給那近處的人。 
    
#18因為我們兩下藉着他被一個聖靈所感，得以進到父面前。
#19這樣，你們不再作外人和客旅，是與聖徒同國，是神家裏的人了； 
#20並且被建造在使徒和先知的根基上，有基督耶穌自己為房角石， 
#21各房靠他聯絡得合式，漸漸成為主的聖殿。 22你們也靠他同被建造，成為神藉着聖靈居住的所在。
    
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
    
===
#出死入生
#1你們死在過犯罪惡之中，他叫你們活過來。 
#2那時，你們在其中行事為人，隨從今世的風俗，順服空中掌權者的首領，就是現今在悖逆之子心中運行的邪靈。 
#3我們從前也都在他們中間，放縱肉體的私慾，隨着肉體和心中所喜好的去行，本為可怒之子，和別人一樣。
    
#4然而，神既有豐富的憐憫，因他愛我們的大愛， 
    
#5當我們死在過犯中的時候，便叫我們與基督一同活過來。你們得救是本乎恩。 
#6他又叫我們與基督耶穌一同復活，一同坐在天上， 
#7要將他極豐富的恩典，就是他在基督耶穌裏向我們所施的恩慈，顯明給後來的世代看。
    
#8你們得救是本乎恩，也因着信；這並不是出於自己，乃是神所賜的； 
#9也不是出於行為，免得有人自誇。 
#10我們原是他的工作，在基督耶穌裏造成的，為要叫我們行善，就是神所預備叫我們行的。


    

