import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 匯入CSV格式的檔案
df = pd.read_csv("pttbeauty2.csv", encoding="utf8")

sns.jointplot(x="comments", y="pushes", data=df)
plt.show()

