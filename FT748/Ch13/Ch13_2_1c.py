import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("HOU_players_stats_2017_18.csv")
df_grouped = df.groupby("Pos")
points = df_grouped["PTS/G"].mean()
rebounds = df_grouped["TRB"].mean()
# 繪出長條圖
index = range(1, 11)
plt.bar(index[0::2], points, label="Points")
plt.bar(index[1::2], rebounds, label="Rebounds")
plt.xticks(index[0::2], points.index)
plt.legend()
plt.ylabel("Points and Rebounds")
plt.xlabel("Position")
plt.title("NBA Houston Rockets") 
plt.show()

