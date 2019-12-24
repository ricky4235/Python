import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset("tips")

sns.set()
sns.distplot(df["total_bill"], rug=True)
plt.show()

