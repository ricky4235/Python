import pandas as pd

df = pd.read_csv("NBA_players_salary_stats_2018.csv")
# 繪製散佈圖
df.plot.scatter(x="PTS", y="salary", 
        title="Scatter Plot of NBA Salary and PTS")
