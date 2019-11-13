import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

print(df.head(2))
# 取得與更新單一純量值
print(df.loc[ordinals[0], "price"])
df.loc[ordinals[0], "price"] = 21.6
print(df.iloc[1,2])
df.iloc[1,2] = 46.3
print(df.head(2))
df.head(2).to_html("Ch12_4_1.html")