from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
import pandas as pd

df = pd.read_csv("Kobe_stats.csv")

output_file("Ch15_5_1.html")

data = ColumnDataSource(data={
        "x": pd.to_datetime(df["Season"]),
        "y": df["PTS"],
        "y1": df["AST"],
        "y2": df["TRB"]
        })
TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

p1 = figure(title="Kobe Bryant的生涯得分", tools=TOOLS,
           title_location="above", x_axis_label="年份",
           y_axis_label="得分",           
           x_axis_type="datetime")
p1.circle(x="x", y="y", source=data, color="red")

p2 = figure(title="Kobe Bryant的生涯助攻和籃板", tools=TOOLS, 
           title_location="above", x_axis_label="年份",
           y_axis_label="助攻和籃板",           
           x_axis_type="datetime")
p2.line(x="x", y="y1", source=data, legend="AST", color="green")
p2.line(x="x", y="y2", source=data, legend="REB", color="blue")

p1.x_range = p2.x_range
layout = gridplot([[p1,p2]])

show(layout)