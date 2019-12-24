import pandas as pd

# 匯入CSV格式的檔案
df = pd.read_csv("products.csv", index_col=0, encoding="utf8")
print(df)
df.to_html("Ch12_2_2b.html")
