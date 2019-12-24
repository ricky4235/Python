import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

print(df.head(3))
# 刪除純量值
print(df.loc[ordinals[0], "price"])
df.loc[ordinals[0], "price"] = None
print(df.iloc[1,2])
df.iloc[1,2] = None
print(df.head(3))
df.head(3).to_html("Ch12_4_2.html")