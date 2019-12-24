from bs4 import BeautifulSoup 

with open("index.html", "r", encoding="utf8") as fp:
    soup = BeautifulSoup(fp, "lxml")
    print(soup)



