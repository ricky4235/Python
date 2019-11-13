import pymysql

# 建立資料庫連接
db = pymysql.connect("localhost", "root", "", "mybooks", charset="utf8")
cursor = db.cursor()  # 建立cursor物件
# 執行SQL指令SELECT
cursor.execute("SELECT * FROM books")
data = cursor.fetchall()   # 取出所有記錄
# 取出查詢結果的每一筆記錄
for row in data:
    print(row[0], row[1])
db.close()  # 關閉資料庫連接

