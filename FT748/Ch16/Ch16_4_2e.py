import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 匯入CSV格式的檔案
df = pd.read_csv("pttbeauty2.csv", encoding="utf8")

sns.pairplot(df, kind="scatter", diag_kind="hist")
plt.show()

sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()