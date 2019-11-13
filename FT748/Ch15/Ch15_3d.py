from bokeh.models.widgets import Dropdown
from bokeh.plotting import output_file, show
from bokeh.layouts import widgetbox

output_file("Ch15_3d.html")

menu = [("setosa","1"),("virginica","2"),("versicolor","3")]

mnu = Dropdown(label="鳶尾花種類", menu=menu)
box = widgetbox(mnu)
show(box)