from bokeh.models.widgets import Select
from bokeh.plotting import output_file, show
from bokeh.layouts import widgetbox

output_file("Ch15_3f.html")

sel = Select(options=["petal", "sepal"], value="petal", title="iris")
box = widgetbox(sel)
show(box)