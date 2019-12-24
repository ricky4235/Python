import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals


print(df)
# 取得與更新整個欄位
print(df.loc[:, "price"])
df.loc[:, "price"] = [23.4, 56.7, 12.1, 90.5, 11.2, 34.1]
print(df.head())
df.head().to_html("Ch12_4_1b.html")
