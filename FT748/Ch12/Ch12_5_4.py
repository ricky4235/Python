import pandas as pd

df = pd.read_csv("products3.csv", encoding="utf8")
df.index = ["A","B","C","D","E","F","G","H","I"]
df.columns = ["type", "name", "price"]

print(df["price"].describe())