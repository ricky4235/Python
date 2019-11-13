import pandas as pd

df = pd.read_csv("duplicated_data.csv")

df1 = df.drop_duplicates("COL_B")
print(df1)
df1.to_html("Ch12_6_2b_01.html") 

df2 = df.drop_duplicates("COL_B", keep="last")
print(df2)
df2.to_html("Ch12_6_2b_02.html") 

df3 = df.drop_duplicates("COL_B", keep=False)
print(df3)
df3.to_html("Ch12_6_2b_03.html") 