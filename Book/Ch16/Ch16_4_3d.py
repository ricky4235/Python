from bokeh.plotting import figure, output_file, show
from bokeh.plotting import ColumnDataSource
from bokeh.models import CategoricalColorMapper
import pandas as pd

# 匯入CSV格式的檔案
df = pd.read_csv("tech_stocks_2017.csv", encoding="utf8")

output_file("Ch16_4_3d.html")

tech_stocks = ["台積電", "鴻海", "廣達", "聯發科", "和碩"]

c_map = CategoricalColorMapper(
           factors=tech_stocks, 
           palette=["blue","green","red","yellow","gray"])

data = ColumnDataSource(data={
        "close": df["Close"],
        "volume": df["Volume"],
        "name": df["Name"]
        })

p = figure(title="蘋概科技股的收盤價與成交量", 
           plot_height=400, plot_width=700, 
           x_range=(min(df.Close), max(df.Close)),
           y_range=(min(df.Volume), max(df.Volume)))
p.diamond(x="close", y="volume", source=data,
          color={"field": "name", "transform": c_map})
p.xaxis.axis_label = "2017年收盤價"
p.yaxis.axis_label = "2017年成交量"

show(p)