import requests
from lxml import html 

r = requests.get("http://www.flag.com.tw/books/school_code_n_algo")
tree = html.fromstring(r.text)

tag_img = tree.cssselect("body > section.allbooks > table > tr:nth-child(2) > td:nth-child(1) > a > img")[0]
print(tag_img)
print(tag_img.tag)
print(tag_img.attrib["src"])

tag_p = tree.cssselect("body > section.allbooks > table > tr:nth-child(2) > td:nth-child(1) > a > p")[0]
print(tag_p)
print(tag_p.tag)
print(tag_p.text_content())


