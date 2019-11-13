import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")
df.index = ["A", "B", "C", "D", "E", "F"]
df.columns = ["type", "name", "price"]

df2 = pd.read_csv("types.csv", encoding="utf8")
df2.index = ["A","B","C","D"]
df2.columns = ["type", "num"]
 
print(df)
print(df2)
df.to_html("Ch12_4_4b_01.html")
df2.to_html("Ch12_4_4b_02.html")

df3 = pd.merge(df, df2)
print(df3)
df3.to_html("Ch12_4_4b_03.html")

df4 = pd.merge(df2, df)
print(df4)
df4.to_html("Ch12_4_4b_04.html")

df5 = pd.merge(df2, df, how='left')
print(df5)
df5.to_html("Ch12_4_4b_05.html")