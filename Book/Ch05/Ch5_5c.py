import requests
from bs4 import BeautifulSoup

url = "http://www.google.com.tw"
r = requests.get(url)
r.encoding = "big5"
soup = BeautifulSoup(r.text, "lxml")
tag_a = soup.find(id="hplogo")
print(tag_a)

