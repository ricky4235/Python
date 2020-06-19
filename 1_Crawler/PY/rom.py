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
#
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
