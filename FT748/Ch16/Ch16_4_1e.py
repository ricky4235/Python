import pandas as pd
import matplotlib.pyplot as plt

# 匯入CSV格式的檔案
df = pd.read_csv("tutsplus2.csv", encoding="utf8")

df["date"] = df["date"].apply(lambda m: m[0:7]) 
df["year"] = df["date"].apply(lambda y: y[0:4])
df = df[df["year"] >= "2016"]
df["date"] = pd.to_datetime(df["date"])
df2 = df.groupby("date").count()

df2["title"].plot(kind="line")
plt.title("Number of Courses per Month")


