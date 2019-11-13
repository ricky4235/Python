from bokeh.plotting import figure, output_file, show
from bokeh.models import CategoricalColorMapper
from bokeh.plotting import ColumnDataSource
from bokeh.layouts import column
import pandas as pd

df = pd.read_csv("iris.csv")
output_file("Ch15_4_2a.html")

c_map = CategoricalColorMapper(
        factors=["setosa","virginica","versicolor"],
        palette=["blue","green","red"]
        )

data = ColumnDataSource(data={
        "x": df["sepal_length"],
        "y": df["sepal_width"],
        "x1": df["petal_length"],
        "y1": df["petal_width"],
        "target": df["target"]
        })

p1 = figure(title="鳶尾花資料集-花萼")
p1.circle(x="x", y="y", source=data, size=15,
         color={"field": "target", "transform": c_map}, 
         legend="target")
p2 = figure(title="鳶尾花資料集-花瓣")
p2.circle(x="x1", y="y1", source=data, size=15,
         color={"field": "target", "transform": c_map}, 
         legend="target")
layout = column(p1, p2)
show(layout)