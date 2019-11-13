import pandas as pd

df = pd.read_csv("missing_data.csv")
# 刪除所有 NaN 的記錄
df1 = df.dropna()
print(df1)
df1.to_html("Ch12_6_1b_01.html")

df2 = df.dropna(how="any")
print(df2)
df2.to_html("Ch12_6_1b_02.html")

df3 = df.dropna(how="all")
print(df3)
df3.to_html("Ch12_6_1b_03.html")

df4 = df.dropna(subset=["COL_B", "COL_C"])
print(df4)
df4.to_html("Ch12_6_1b_04.html")