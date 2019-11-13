import seaborn as sns

df = sns.load_dataset("tips")
print(df.head())
df.head().to_html("Ch14_1_4.html")


