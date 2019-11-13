from bokeh.models import Slider, ColumnDataSource
from bokeh.io import curdoc
from bokeh.layouts import column
from bokeh.plotting import figure
import random

num_of_points = 100
data = ColumnDataSource(data = {
        "x": random.sample(range(0,600),num_of_points),
        "y": random.sample(range(0,600),num_of_points)
        })

p = figure(title="Random Scatter Plot")
p.circle(x="x", y="y", source=data, color="blue")
sld = Slider(start=0, end=500, step=10, value=num_of_points,
             title="Slide to Increase Number of Points")

def callback(attr, old, new):
    points = sld.value
    data.data = {"x": random.sample(range(0,600),points),
                 "y": random.sample(range(0,600),points)
                 }
sld.on_change("value", callback)

layout = column(sld, p)
curdoc().add_root(layout)