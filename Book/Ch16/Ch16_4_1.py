import pandas as pd

# 匯入CSV格式的檔案
df = pd.read_csv("tutsplus.csv", encoding="utf8")

print(df.info())

print(df.head())
df.head().to_html("Ch16_4_1.html")



