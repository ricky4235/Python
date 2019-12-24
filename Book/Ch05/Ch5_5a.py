import urllib.request

url = "http://hueyanchen.myweb.hinet.net/fchart05.png"
response = urllib.request.urlopen(url)
fp = open("fchart06.png", "wb")
size = 0
while True:
    info = response.read(10000)
    if len(info) < 1:
        break
    size = size + len(info)
    fp.write(info)    
print(size, "個字元下載...")
fp.close()
response.close()

