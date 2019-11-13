import requests

r = requests.get("http://hueyanchen.myweb.hinet.net/test.html")
r.encoding = "utf-8"

fp = open("test.txt", "w", encoding="utf8")
fp.write(r.text)
print("寫入檔案test.txt...")
fp.close()

