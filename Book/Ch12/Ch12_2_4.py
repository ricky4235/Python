import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

for index, row in df.iterrows() :
    print(index, row["分類"], row["商店"], row["價格"])