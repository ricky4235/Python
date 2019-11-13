# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 13:40:26 2019

@author: 11004076
"""

import requests
from bs4 import BeautifulSoup

r = requests.get('https://www.google.com/search?q=python3')
soup = BeautifulSoup(r.text, 'lxml')

a_tags = soup.select('div.r a')
for t in a_tags:
    print(t.text)