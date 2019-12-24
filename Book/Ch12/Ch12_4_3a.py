import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

df["sales"] = [124.5,227.5,156.7,435.6,333.7,259.8] 
print(df.head())
df.head().to_html("Ch12_4_3a_01.html")
df.loc[:,"city"] = ["台北","新竹","台北","台中","新北","高雄"]
print(df.head())
df.head().to_html("Ch12_4_3a_02.html")