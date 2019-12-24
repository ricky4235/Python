import pandas as pd

df = pd.read_csv("duplicated_data.csv")

df1 = df.drop_duplicates()
print(df1)
df1.to_html("Ch12_6_2a.html") 
