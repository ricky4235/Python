from bokeh.models.widgets import Slider
from bokeh.plotting import output_file, show
from bokeh.layouts import widgetbox

output_file("Ch15_3e.html")

sld = Slider(start=0, end=50, value=25,
             title="輸入0~50", step=5)
box = widgetbox(sld)
show(box)