import pandas as pd

df = pd.read_csv("missing_data.csv")
# 填補遺失資料
df1 = df.fillna(value=1)
print(df1)
df1.to_html("Ch12_6_1c_01.html")

df["COL_B"] = df["COL_B"].fillna(df["COL_B"].mean())
print(df)
df.to_html("Ch12_6_1c_02.html")

df["COL_C"] = df["COL_C"].fillna(df["COL_C"].median())
print(df)
df.to_html("Ch12_6_1c_03.html")
