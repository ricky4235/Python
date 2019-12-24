import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("HOU_players_stats_2017_18.csv")
df_grouped = df.groupby("Pos")
points = df_grouped["PTS/G"].mean()
rebounds = df_grouped["TRB"].mean()
# 繪出長條圖
plt.bar([1, 2, 3, 4, 5], points, label="Points")
plt.bar([1, 2, 3, 4, 5], rebounds, label="Rebounds")
plt.xticks([1, 2, 3, 4, 5], points.index)
plt.legend()
plt.ylabel("Points and Rebounds")
plt.xlabel("Position")
plt.title("NBA Houston Rockets") 
plt.show()

