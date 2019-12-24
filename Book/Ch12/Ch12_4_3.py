import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

print(df.tail(3))
# 新增記錄
df.loc["G"] = ["科學", "全聯超", 28.5]
print(df.tail(3))
df.tail(3).to_html("Ch12_4_3_01.html")
s = pd.Series({"type":"科學","name":"大潤發","price":79.2})
df2 = df.append(s, ignore_index=True)
print(df2.tail(3))
df2.tail(3).to_html("Ch12_4_3_02.html")