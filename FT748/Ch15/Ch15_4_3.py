from bokeh.plotting import figure, output_file, show
from bokeh.models import CategoricalColorMapper
from bokeh.plotting import ColumnDataSource
from bokeh.layouts import column
from bokeh.models.widgets import Dropdown, Tabs, Panel
import pandas as pd

df = pd.read_csv("iris.csv")
output_file("Ch15_4_3.html")

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
menu = [("setosa","1"),("virginica","2"),("versicolor","3")]
mnu1 = Dropdown(label="鳶尾花種類", menu=menu)
mnu2 = Dropdown(label="鳶尾花種類", menu=menu)

tab1 = Panel(child=column(mnu1,p1), title="花萼")
tab2 = Panel(child=column(mnu2,p2), title="花瓣")
tabs = Tabs(tabs=[tab1, tab2])

show(tabs)