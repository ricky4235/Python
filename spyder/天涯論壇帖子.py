# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 15:16:49 2019

@author: 11004076
"""

"""
使用requests_html
"""

from requests_html import HTMLSession
import io

session = HTMLSession()



# 爬取天涯论坛帖子
url = 'http://bbs.tianya.cn/post-culture-488321-1.shtml'
r = session.get(url)
# 楼主名字
author = r.html.find('div.atl-info span a', first=True).text
# 总页数
div = r.html.find('div.atl-pages', first=True)
links = div.find('a')
total_page = 1 if links == [] else int(links[-2].text)
# 标题
title = r.html.find('span.s_title span', first=True).text

with io.open(f'{title}.txt', 'x', encoding='utf-8') as f:
    for i in range(1, total_page + 1):
        s = url.rfind('-')
        r = session.get(url[:s + 1] + str(i) + '.shtml')
        # 从剩下的里面找楼主的帖子
        items = r.html.find(f'div.atl-item[_host={author}]')
        for item in items:
            content: str = item.find('div.bbs-content', first=True).text
            # 去掉回复
            if not content.startswith('@'):
                f.write(content + "\n")