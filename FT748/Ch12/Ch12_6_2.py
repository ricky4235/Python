import pandas as pd

df = pd.read_csv("duplicated_data.csv")
print(df)
df.to_html("Ch12_6_2.html")
