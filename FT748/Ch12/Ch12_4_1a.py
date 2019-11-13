import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

print(df.head(3))
# 取得與更新單筆記錄
print(df.loc[ordinals[1]])
s = ["居家", "家樂福", 30.4] 
df.loc[ordinals[1]] = s
print(df.head(3))
df.head(3).to_html("Ch12_4_1a.html")