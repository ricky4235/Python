from bokeh.plotting import figure, output_file, show
from bokeh.plotting import ColumnDataSource
from bokeh.models import CategoricalColorMapper
from bokeh.models import HoverTool
import pandas as pd

df = pd.read_csv("iris.csv")
output_file("Ch15_5_2a.html")

c_map = CategoricalColorMapper(
        factors=["setosa","virginica","versicolor"],
        palette=["blue","green","red"]
        )
hover_tool = HoverTool(tooltips = [
             ("花瓣長度", "@petal_length"),
             ("花瓣寬度", "@petal_width"),
             ("種類", "@target")        
             ])
data = ColumnDataSource(df)

p = figure(title="鳶尾花資料集", tools=["box_select", hover_tool])

p.circle(x="petal_length", y="petal_width", source=data, size=15,
         color={"field": "target", "transform": c_map}, legend="target",
         selection_color="green", nonselection_fill_alpha=0.3, 
         nonselection_fill_color="grey")

show(p)