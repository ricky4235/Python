import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

print(df.index)
print(df.columns)
print(df.values)  