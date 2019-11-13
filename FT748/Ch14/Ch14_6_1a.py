import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.set()
sns.regplot(x=df["total_bill"], y=df["tip"])
sns.lmplot(x="total_bill", y="tip", data=df)
plt.show()


