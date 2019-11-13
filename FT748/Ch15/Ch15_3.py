from bokeh.models.widgets import Button
from bokeh.plotting import output_file, show
from bokeh.layouts import widgetbox

output_file("Ch15_3.html")

btn = Button(label="下一頁")
box = widgetbox(btn)
show(box)