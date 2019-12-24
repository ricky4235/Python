import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

print(df.head(3))
# 刪除欄位
df2 = df.drop(["price"], axis=1)
print(df2.head(3))
df2.head(3).to_html("Ch12_4_2b.html")