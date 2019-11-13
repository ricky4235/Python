import requests
from lxml import html 

r = requests.get("http://www.flag.com.tw/books/school_code_n_algo")
tree = html.fromstring(r.text)
print(tree)

for ele in tree.getchildren():
    print(ele)


