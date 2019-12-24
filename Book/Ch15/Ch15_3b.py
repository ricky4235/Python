from bokeh.models.widgets import CheckboxGroup
from bokeh.plotting import output_file, show
from bokeh.layouts import widgetbox

output_file("Ch15_3b.html")

ckb = CheckboxGroup(labels=["setosa","virginica","versicolor"], 
                    active=[1, 2])
box = widgetbox(ckb)
show(box)