import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

df2 = df.set_index("price")
print(df2)
df2.to_html("Ch12_3_3_01.html")

df2.sort_index(ascending=False, inplace=True)
print(df2)
df2.to_html("Ch12_3_3_02.html")