from bokeh.models.widgets import TextInput
from bokeh.plotting import output_file, show
from bokeh.layouts import widgetbox

output_file("Ch15_3a.html")

txt = TextInput(title="請輸入最大值:", value="100")
box = widgetbox(txt)
show(box)