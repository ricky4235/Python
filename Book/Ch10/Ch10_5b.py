import pymysql

d = {
   "id": "P0005",
   "title": "Node.js程式設計",
   "author": "陳會安",
   "price": 650,
   "cat": "程式設計",
   "date": "2018-02-01"
}

# 建立資料庫連接
db = pymysql.connect("localhost", "root", "", "mybooks", charset="utf8")
cursor = db.cursor()  # 建立cursor物件
# 建立SQL指令INSERT字串
sql = """INSERT INTO books (id,title,author,price,category,pubdate)
         VALUES ('{0}','{1}','{2}',{3},'{4}','{5}')"""
sql = sql.format(d['id'],d['title'],d['author'],d['price'],d['cat'],d['date'])
print(sql)
try:
    cursor.execute(sql)   # 執行SQL指令
    db.commit() # 確認交易
    print("新增一筆記錄...")
except:
    db.rollback() # 回復交易 
    print("新增記錄失敗...")
db.close()  # 關閉資料庫連接
