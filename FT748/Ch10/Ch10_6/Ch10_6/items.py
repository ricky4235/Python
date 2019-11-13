import scrapy

class QuoteItem(scrapy.Item):
    # 定義Item的欄位
    quote = scrapy.Field()
    author = scrapy.Field()

