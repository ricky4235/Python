import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("HOU_players_stats_2017_18.csv")
df_grouped = df.groupby("Pos")
points = df_grouped["PTS/G"].mean()
rebounds = df_grouped["TRB"].mean()
data = pd.DataFrame()
data["Points"] = points
data["Rebounds"] = rebounds
# 繪出長條圖
print(points)
points.plot.bar()
plt.title("Points")
print(data)
data.plot.bar()
plt.title("Points and Rebounds")