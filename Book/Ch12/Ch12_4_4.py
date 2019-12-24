import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

columns = ["type", "name", "price"]
# 建立空的DataFrame物件
df_empty = pd.DataFrame(None, index=ordinals, columns=columns)
print(df_empty)
# 複製DataFrame物件
df_copy = df.copy()
print(df_copy)