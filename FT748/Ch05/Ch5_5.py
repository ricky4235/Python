import requests

url = "http://hueyanchen.myweb.hinet.net/fchart05.png"
path = "fchart05.png"
response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(path, 'wb') as fp:
        for chunk in response:
            fp.write(chunk)
    print("圖檔已經下載")        
else:
    print("錯誤! HTTP請求失敗...")
    