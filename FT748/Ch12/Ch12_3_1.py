import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

print(df["price"].head(3))
print(df[["type","name"]].head(3))
df[["type","name"]].head(3).to_html("Ch12_3_1.html")

print(df.price.head(3))   # 使用屬性方式
