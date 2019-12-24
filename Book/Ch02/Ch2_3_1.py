import requests

r = requests.get("http://hueyanchen.myweb.hinet.net/test.html")

print(r.text)
print(r.encoding)

r = requests.get("http://hueyanchen.myweb.hinet.net/test.html")
r.encoding = 'utf-8'

print(r.text)
print(r.encoding)