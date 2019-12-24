import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

print("資料數= ", len(df))
print("形狀= ", df.shape)  
df.info()