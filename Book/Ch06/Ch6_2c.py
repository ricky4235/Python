import requests
from lxml import html 

r = requests.get("http://www.flag.com.tw/books/school_code_n_algo")
tree = html.fromstring(r.text)

tag_img = tree.xpath("/html/body/section[2]/table/tr[2]/td[1]/a/img")[0]
print(tag_img.tag)
print(tag_img.getparent().tag)
print(tag_img.getnext().tag)
print("------------------")
tag_p = tree.xpath("/html/body/section[2]/table/tr[2]/td[1]/a/p")[0]
print(tag_p.tag)
print(tag_p.getprevious().tag)
