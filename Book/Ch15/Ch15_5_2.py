from bokeh.plotting import figure, output_file, show
from bokeh.plotting import ColumnDataSource
from bokeh.models import HoverTool
import pandas as pd

df = pd.read_csv("Kobe_stats.csv")
data = pd.DataFrame()
data["Season"] = pd.to_datetime(df["Season"])
data["PTS"] = df["PTS"]
data["AST"] = df["AST"]
data["REB"] = df["TRB"]
output_file("Ch15_5_2.html")

hover_tool = HoverTool(tooltips = [
             ("得分", "@PTS"),
             ("助攻", "@AST"),
             ("籃板", "@REB")        
             ])

data2 = ColumnDataSource(data)

p = figure(title="Kobe Bryant的生涯得分、助攻和籃板", 
           title_location="above", x_axis_label="年份",
           y_axis_label="得分、助攻和籃板",           
           x_axis_type="datetime", tools=[hover_tool])
p.line(x="Season", y="PTS", source=data2,
       legend="PTS", color="red")
p.line(x="Season", y="AST", source=data2,
       legend="AST", color="green")
p.line(x="Season", y="REB", source=data2,
       legend="REB", color="blue")

show(p)