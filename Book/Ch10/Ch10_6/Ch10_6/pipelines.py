import pymysql
import datetime

class MysqlPipeline(object):
    def __init__(self):
        self.db = pymysql.connect("localhost","root","","myquotes",
                                  charset="utf8")
        
    def open_spider(self, spider):    
        self.cursor = self.db.cursor();  # 建立cursor物件
        
    def process_item(self, item, spider):
        # 建立SQL指令INSERT字串
        sql = """INSERT INTO quotes(quote,author,createDate) 
                 VALUE(%s,%s,%s)"""
        try:
            self.cursor.execute(sql,
                (item["quote"],
                 item["author"],
                 datetime.datetime.now()
                         .strftime('%Y-%m-%d %H:%M:%S')
                 ))   # 執行SQL指令
            self.db.commit()   # 確認交易
        except Exception as err:
            self.db.rollback() # 回復交易 
            print("錯誤! 插入記錄錯誤...", err)
        return item

    def close_spider(self, spider):            
        self.db.close()  # 關閉資料庫連接
