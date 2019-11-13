import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

df2 = df.sort_values("price", ascending=False)
print(df2)
df2.to_html("Ch12_3_3a_01.html")

df.sort_values(["type","price"], inplace=True)
print(df)
df.to_html("Ch12_3_3a_02.html")