import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("GSW_players_stats_2017_18.csv")
df_grouped = df.groupby("Pos")
position = df_grouped["Pos"].count()
explode = (0, 0, 0.2, 0, 0.2)
# 繪出派圖
patches, texts = plt.pie(position, 
                         labels=position.index,
                         explode=explode)
plt.legend(patches, position.index, loc="best")
plt.axis("equal")
plt.title("NBA Golden State Warriors") 
plt.show()
