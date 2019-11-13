import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

print(df.loc[ordinals[1]])
print(type(df.loc[ordinals[1]]))
print(df.loc[:,["name","price"]])
print(df.loc[["C","F"], ["name","price"]])

df.loc[:,["name","price"]].to_html("Ch12_3_1b_01.html")
df.loc[["C","F"], ["name","price"]].to_html("Ch12_3_1b_02.html")

print(df.loc["C":"E", ["name","price"]])
print(df.loc["C", ["name","price"]])

df.loc["C":"E", ["name","price"]].to_html("Ch12_3_1b_03.html")
# 取得單一純量值
print(df.loc[ordinals[0], "name"])
print(type(df.loc[ordinals[0],"name"]))
print(df.loc["A", "price"])
print(type(df.loc["A", "price"]))

print(df.loc[ordinals[0]]["name"])
print(df.loc["A"]["price"])
