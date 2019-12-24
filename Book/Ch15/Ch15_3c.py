from bokeh.models.widgets import RadioGroup
from bokeh.plotting import output_file, show
from bokeh.layouts import widgetbox

output_file("Ch15_3c.html")

rdb = RadioGroup(labels=["setosa","virginica","versicolor"], 
                    active=1)
box = widgetbox(rdb)
show(box)