import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")
columns = ["type", "name", "price"]
df.index = ["A", "B", "C", "D", "E", "F"]
df.columns = columns

df2 = pd.read_csv("products2.csv", encoding="utf8")
df2.index = ["A","B","C"]
df2.columns = columns
 
print(df)
print(df2)
df.to_html("Ch12_4_4a_01.html")
df2.to_html("Ch12_4_4a_02.html")
  
df3 = pd.concat([df,df2])  
print(df3)
df3.to_html("Ch12_4_4a_03.html")

df4 = pd.concat([df,df2], ignore_index=True)
print(df4) 
df4.to_html("Ch12_4_4a_04.html") 
df4.to_csv("products3.csv")