import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("NBA_players_salary_stats_2018.csv")
# 繪製散佈圖
plt.scatter(df["AST"], df["salary"])
plt.ylabel("Salary")
plt.xlabel("AST")
plt.title("Scatter Plot of NBA Salary and AST") 
plt.show()