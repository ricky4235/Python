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
 #20就是照他在基督身上所運行的大能大力，使他從死裏復活，叫他在天上坐在自己的右邊， 
 #21遠超過一切執政的、掌權的、有能的、主治的，和一切有名的；
     #不但是今世的，連來世的也都超過了。 
 #22又將萬有服在他的腳下，使他為教會作萬有之首。 
 #23教會是他的身體，是那充滿萬有者所充滿的。
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
        

 #15因此，我既聽見你們 信從主耶穌，親愛眾聖徒， 
 #16就為你們不住地感謝神。禱告的時候，常提到你們， 
 #17求我們主耶穌基督的神，榮耀的父，將那賜人智慧和啟示的靈賞給你們，使你們真知道他， 
 #18並且照明你們心中的眼睛，
 #              使你們知道他 的恩召  有何等  指望， 
 #              他在聖徒中得 的基業  有何等  豐盛的榮耀； 
 #19並知道他向我們這信的人所顯的能力  是何等浩大， 
 
 #20就是照他在基督身上所運行的大能大力，使他從死裏復活，叫他在天上坐在自己的右邊， 
 #21遠超過一切執政的、掌權的、有能的、主治的，和一切有名的；
     #不但是今世的，連來世的也都超過了。 
 #22又將萬有服在他的腳下，使他為教會作萬有之首。 
 #23教會是他的身體，是那充滿萬有者所充滿的。
