import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("GSW_players_stats_2017_18.csv")
df_grouped = df.groupby("Pos")
position = df_grouped["Pos"].count()
# 繪出長條圖
plt.barh([1, 2, 3, 4, 5], position)
plt.yticks([1, 2, 3, 4, 5], position.index)
plt.xlabel("Number of People")
plt.ylabel("Position")
plt.title("NBA Golden State Warriors") 
plt.show()

