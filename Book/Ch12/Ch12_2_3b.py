import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
print(df.head(3)) 
df.head(3).to_html("Ch12_2_3b.html")