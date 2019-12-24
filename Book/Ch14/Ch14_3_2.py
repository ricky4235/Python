import seaborn as sns

df = sns.load_dataset("iris")
print(df.head())
df.head().to_html("Ch14_3_2.html")
