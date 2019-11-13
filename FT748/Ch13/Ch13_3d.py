import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("GSW_players_stats_2017_18.csv")
df_grouped = df.groupby("Pos")
position = df_grouped["Pos"].count()
explode = (0, 0, 0.2, 0, 0.2)
# 繪出派圖
position.plot(kind="pie",
              figsize=(6, 6),
              explode=explode, 
              title="NBA Golden State Warriors") 
plt.legend(position.index, loc="best")