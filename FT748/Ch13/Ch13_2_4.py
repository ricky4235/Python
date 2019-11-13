import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("NBA_players_salary_stats_2018.csv")
# 繪製散佈圖
plt.scatter(df["PTS"], df["salary"])
plt.ylabel("Salary")
plt.xlabel("PTS")
plt.title("Scatter Plot of NBA Salary and PTS") 
plt.show()