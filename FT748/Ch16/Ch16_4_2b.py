import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 匯入CSV格式的檔案
df = pd.read_csv("pttbeauty2.csv", encoding="utf8")

sns.distplot(df["images"], kde=False)
plt.title("Number of Images")
plt.xlabel("Number of Images")
plt.ylabel("Number of Posts")
plt.show()

