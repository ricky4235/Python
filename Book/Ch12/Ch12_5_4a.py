import pandas as pd

df = pd.read_csv("products3.csv", encoding="utf8")
df.index = ["A","B","C","D","E","F","G","H","I"]
df.columns = ["type", "name", "price"]

print("總和: ", df["price"].sum())
print("平均: ", df["price"].mean())
print("最大: ", df["price"].max())
print("最小: ", df["price"].min())
print("計數: ", df["price"].count())
print("標準差: ", df["price"].std())
print("變異數: ", df["price"].var())
print("25%: ", df["price"].quantile(q=0.25))
print("50%: ", df["price"].quantile(q=0.5))
print("75%: ", df["price"].quantile(q=0.75))
