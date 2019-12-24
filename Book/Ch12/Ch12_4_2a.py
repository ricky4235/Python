import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

print(df.head())
# 刪除記錄
df2 = df.drop(["B", "D"])    # 2,4 筆
print(df2.head())
df2.head().to_html("Ch12_4_2a_01.html")
df.drop(df.index[[2,3]], inplace=True) # 3,4 筆
print(df.head())
df.head().to_html("Ch12_4_2a_02.html")