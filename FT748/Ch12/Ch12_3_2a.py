import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

print(df[(df.price > 15) & (df.price < 25)])
df[(df.price > 15) & (df.price < 25)].to_html("Ch12_3_2a_01.html")

df.loc["G"] = ["科學", "全聯超", 28.5]
print(df[df["type"].str.startswith("科")])
df[df["type"].str.startswith("科")].to_html("Ch12_3_2a_02.html")
