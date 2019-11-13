import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("NBA_salary_rankings_2018.csv")

df.boxplot(column="salary",
             by="pos",
             figsize=(6,5))

plt.xticks(rotation=25)
plt.title("Box Plot of NBA Salary") 
