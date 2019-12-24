# -*- coding: utf-8 -*-
import scrapy
import re
from Ch9_4.items import TutsplusItem

class TutsplusSpider(scrapy.Spider):
    name = 'tutsplus'
    allowed_domains = ['code.tutsplus.com']
    start_urls = ['https://code.tutsplus.com/tutorials']

    def parse(self, response):
        # 取得目前頁面所有的超連結
        links = response.xpath('//a/@href').extract()
        
        crawledLinks = []
        # 取出符合條件的超連結, 即其他頁面
        linkPattern = re.compile("^\/tutorials\?page=\d+")
        for link in links:
            if linkPattern.match(link) and not link in crawledLinks:
                link = "http://code.tutsplus.com" + link
                crawledLinks.append(link)
                yield scrapy.Request(link, self.parse)
 
        # 取得每一頁的詳細課程資訊
        for tut in response.css("li.posts__post"):
            item = TutsplusItem()
            
            item["title"] = tut.css(".posts__post-title > h1::text").extract_first()
            item["author"] = tut.css(".posts__post-author-link::text").extract_first()
            item["category"] = tut.css(".posts__post-primary-category-link::text").extract_first()
            item["date"] = tut.css(".posts__post-publication-date::text").extract_first()
            yield item
