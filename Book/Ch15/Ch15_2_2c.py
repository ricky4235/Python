from bokeh.plotting import figure, output_file, show
from bokeh.models import CategoricalColorMapper
import pandas as pd

df = pd.read_csv("iris.csv")
output_file("Ch15_2_2c.html")

c_map = CategoricalColorMapper(
        factors=["setosa","virginica","versicolor"],
        palette=["blue","green","red"]
        )
p = figure(title="鳶尾花資料集")

p.circle(x="sepal_length", y="sepal_width", source=df, size=15,
         color={"field": "target", "transform": c_map})

show(p)