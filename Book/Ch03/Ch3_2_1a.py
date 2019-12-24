import requests 
from bs4 import BeautifulSoup

r = requests.get("http://hueyanchen.myweb.hinet.net")
r.encoding = "utf8"
soup = BeautifulSoup(r.text, "lxml")
print(soup)



