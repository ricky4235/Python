import requests 

url = "https://api.github.com/user"

r = requests.get(url, auth=('hueyan@ms2.hinet.net', '********'))
if r.status_code == requests.codes.ok:
    print(r.headers['Content-Type'])
    print(r.json())
else:
    print("HTTP請求錯誤...")