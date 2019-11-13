import pandas as pd

df1 = pd.read_csv("stocks\\2330.TW.csv", encoding="utf8")
df1 = df1.dropna()
df1["Name"] = "台積電"
df2 = pd.read_csv("stocks\\2317.TW.csv", encoding="utf8")
df2 = df2.dropna()
df2["Name"] = "鴻海"
df3 = pd.read_csv("stocks\\2382.TW.csv", encoding="utf8")
df3 = df3.dropna()
df3["Name"] = "廣達"
df4 = pd.read_csv("stocks\\2454.TW.csv", encoding="utf8")
df4 = df4.dropna()
df4["Name"] = "聯發科"
df5 = pd.read_csv("stocks\\4938.TW.csv", encoding="utf8")
df5 = df5.dropna()
df5["Name"] = "和碩"

data = pd.concat([df1, df2, df3, df4, df5])

print(data.head())
data.head().to_html("Ch16_4_3c.html")
print(data.info())

data.to_csv("tech_stocks_2017.csv", index=False, encoding="utf8")
print("存入tech_stocks_2017.csv")



