from bokeh.plotting import figure
from bokeh.models import CategoricalColorMapper, Select
from bokeh.plotting import ColumnDataSource
from bokeh.layouts import column
from bokeh.io import curdoc
import pandas as pd

df = pd.read_csv("iris.csv")

c_map = CategoricalColorMapper(
        factors=["setosa","virginica","versicolor"],
        palette=["blue","green","red"]
        )

data = ColumnDataSource(data={
        "x": df["petal_length"],
        "y": df["petal_width"],
        "target": df["target"]
        })

p = figure(title="IRIS DataSet")
p.circle(x="x", y="y", source=data, size=15,
         color={"field": "target", "transform": c_map}, 
         legend="target")

sel = Select(options=["petal", "sepal"], value="petal", title="iris")
def callback(attr, old, new):
    if sel.value == "petal":
        data.data = {
              "x": df["petal_length"],
              "y": df["petal_width"],
              "target": df["target"]
              }
    else:
        data.data = {
              "x": df["sepal_length"],
              "y": df["sepal_width"],
              "target": df["target"]
              }        
sel.on_change("value", callback)

layout = column(sel, p)
curdoc().add_root(layout)