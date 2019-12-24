import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.set()
sns.kdeplot(df["total_bill"])
sns.kdeplot(df["total_bill"], bw=2, label="bw: 2")
sns.kdeplot(df["total_bill"], bw=5, label="bw: 5")
plt.show()

