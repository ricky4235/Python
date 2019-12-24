import scrapy
from Ch10_6.items import QuoteItem

class QuotesSpider(scrapy.Spider):
    name = 'quotes'
    allowed_domains = ['quotes.toscrape.com']
    start_urls = ['http://quotes.toscrape.com/']

    def parse(self, response):
        for quote in response.css("div.quote"):
            item = QuoteItem()
            item["quote"] = quote.css("span.text::text").extract_first()
            item["author"] = quote.xpath(".//small/text()").extract_first()
            yield item
            
        nextPg = response.xpath("//li[@class='next']/a/@href").extract_first()
        if nextPg is not None:
            nextPg = response.urljoin(nextPg)
            yield scrapy.Request(nextPg, callback=self.parse)
