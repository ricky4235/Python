from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
import pandas as pd

df = pd.read_csv("Kobe_stats.csv")

output_file("Ch15_5_1a.html")

data = ColumnDataSource(data={
        "x": pd.to_datetime(df["Season"]),
        "y1": df["AST"],
        "y2": df["TRB"]
        })
TOOLS = "pan,wheel_zoom,box_zoom,reset,save,box_select,lasso_select"

p1 = figure(title="Kobe Bryant的生涯助攻", tools=TOOLS,
           title_location="above", x_axis_label="年份",
           y_axis_label="助攻",           
           x_axis_type="datetime")
p1.line(x="x", y="y1", source=data, color="red")

p2 = figure(title="Kobe Bryant的生涯籃板", tools=TOOLS, 
           title_location="above", x_axis_label="年份",
           y_axis_label="籃板",           
           x_axis_type="datetime")
p2.line(x="x", y="y2", source=data, color="blue")

p1.y_range = p2.y_range
layout = gridplot([[p1,p2]])

show(layout)