import pandas as pd

df = pd.read_csv("missing_data.csv")
print(df)
df.to_html("Ch12_6_1.html")