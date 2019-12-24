import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

print(df[0:3])     # 不含 3
print(df["C":"E"]) # 含 "E"
df[0:3].to_html("Ch12_3_1a_01.html")
df["C":"E"].to_html("Ch12_3_1a_02.html")