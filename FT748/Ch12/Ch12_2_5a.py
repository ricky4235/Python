import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df2 = df.set_index(["分類", "商店"])
df2.sort_index(ascending=False, inplace=True)
print(df2)
df2.to_html("Ch12_2_5a.html")