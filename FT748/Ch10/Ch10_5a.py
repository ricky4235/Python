import pymysql

book = "P0004,Python程式設計,陳會安,550,程式設計,2018-01-01"
f = book.split(",")

# 建立資料庫連接
db = pymysql.connect("localhost", "root", "", "mybooks", charset="utf8")
cursor = db.cursor()  # 建立cursor物件
# 建立SQL指令INSERT字串
sql = """INSERT INTO books (id,title,author,price,category,pubdate)
         VALUES ('{0}','{1}','{2}',{3},'{4}','{5}')"""
sql = sql.format(f[0], f[1], f[2], f[3], f[4], f[5])
print(sql)
try:
    cursor.execute(sql)   # 執行SQL指令
    db.commit() # 確認交易
    print("新增一筆記錄...")
except:
    db.rollback() # 回復交易 
    print("新增記錄失敗...")
db.close()  # 關閉資料庫連接

