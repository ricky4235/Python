import requests
from bs4 import BeautifulSoup

r = requests.get("http://hueyanchen.myweb.hinet.net/test.html")
r.encoding = "utf-8"
soup = BeautifulSoup(r.text, "lxml")

fp = open("test2.txt", "w", encoding="utf8")
fp.write(soup.prettify())
print("寫入檔案test2.txt...")
fp.close()



