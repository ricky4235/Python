import pandas as pd
import pymysql

# 建立資料庫連接
db = pymysql.connect("localhost", "root", "", "mybooks", charset="utf8")
# SQL指令SELECT
sql = "SELECT * FROM books"
# 從資料庫匯入資料
df = pd.read_sql(sql, db)
print(df.head())
df.to_html("Ch12_2_2c.html")
db.close()  # 關閉資料庫連接

