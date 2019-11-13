import requests

r = requests.get("http://hueyanchen.myweb.hinet.net/test.html")
r.encoding = 'utf-8'
print(r.text)
print("----------------------")

r = requests.get("http://hueyanchen.myweb.hinet.net/test.html")
r.encoding = 'utf-8'
print(r.content)
print("----------------------")

r = requests.get("http://hueyanchen.myweb.hinet.net/test.html", stream=True)
r.encoding = 'utf-8'
print(r.raw)
print(r.raw.read(15))

