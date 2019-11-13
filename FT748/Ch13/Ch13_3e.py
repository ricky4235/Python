import pandas as pd

df = pd.read_csv("Kobe_stats.csv")
# 繪製折線圖
data = pd.DataFrame()
data["Season"] = pd.to_datetime(df["Season"])
data["PTS"] = df["PTS"]
data["AST"] = df["AST"]
data["REB"] = df["TRB"]
data = data.set_index("Season")

data.plot(kind="line")
