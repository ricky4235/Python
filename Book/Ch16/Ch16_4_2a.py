import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 匯入CSV格式的檔案
df = pd.read_csv("pttbeauty2.csv", encoding="utf8")

sns.distplot(df["pushes"], kde=False)
plt.title("Number of Pushes")
plt.xlabel("Number of Pushes")
plt.ylabel("Number of Posts")
plt.show()

