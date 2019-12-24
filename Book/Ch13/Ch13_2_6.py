import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("Kobe_stats.csv")
# 繪製折線圖
df["Season"] = pd.to_datetime(df["Season"])
df = df.set_index("Season")
#print(df)
plt.plot(df["PTS"], "r-o", label="PTS")
plt.plot(df["AST"], "b-o", label="AST")
plt.plot(df["TRB"], "g-o", label="REB")
plt.legend()
plt.ylabel("Stats")
plt.xlabel("Season")
plt.title("Kobe Bryant") 
plt.show()