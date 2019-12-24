import pandas as pd
import matplotlib.pyplot as plt

# 匯入CSV格式的檔案
df = pd.read_csv("tutsplus.csv", encoding="utf8")

print(df["author"].value_counts().head(5))

df["author"].value_counts().head(5).plot(kind="barh")
plt.title("Top 5 Authors")
