from bokeh.plotting import figure, output_file, show
import pandas as pd
from datetime import datetime

df = pd.read_csv("Kobe_stats.csv")
data = pd.DataFrame()
data["Season"] = pd.to_datetime(df["Season"])
data["PTS"] = df["PTS"]
data["AST"] = df["AST"]
data["REB"] = df["TRB"]
output_file("Ch15_2_2b.html")

p = figure(title="Kobe Bryant的生涯得分、助攻和籃板", 
           title_location="above", x_axis_label="年份",
           y_axis_label="得分、助攻和籃板",           
           x_axis_type="datetime", y_range=(0, 40),
           x_range=(datetime(1995,1,1),datetime(2016,1,1)))
p.line(data["Season"], data["PTS"], legend="PTS", color="red")
p.line(data["Season"], data["AST"], legend="AST", color="green")
p.line(data["Season"], data["REB"], legend="REB", color="blue")

show(p)