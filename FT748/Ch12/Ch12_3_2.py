import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

print(df[df.price > 20])
df[df.price > 20].to_html("Ch12_3_2_01.html")

print(df[df["type"].isin(["科技","居家"])])
df[df["type"].isin(["科技","居家"])].to_html("Ch12_3_2_02.html")
