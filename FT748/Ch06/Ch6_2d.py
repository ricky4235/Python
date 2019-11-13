import requests
from lxml import html 

r = requests.get("http://www.flag.com.tw/books/school_code_n_algo")
tree = html.fromstring(r.text)

tag_img = tree.xpath("/html/body/section[2]/table/tr[2]/td[1]/a/img")[0]

for ele in tag_img.getparent().getchildren():
    print(ele.tag)
