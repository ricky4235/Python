import pandas as pd
import matplotlib.pyplot as plt

# 匯入CSV格式的檔案
df = pd.read_csv("tutsplus.csv", encoding="utf8")

print(df["category"].value_counts().head(10))

df["category"].value_counts().head(10).plot(kind="barh")
plt.title("Top 10 Categories")
