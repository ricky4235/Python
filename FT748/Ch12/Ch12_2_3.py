import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

print(df.head()) 
print(df.head(3))
df.head().to_html("Ch12_2_3_01.html")
df.head(3).to_html("Ch12_2_3_02.html")