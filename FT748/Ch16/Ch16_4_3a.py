from bokeh.plotting import figure, output_file, show
from bokeh.plotting import ColumnDataSource
import pandas as pd

# 匯入CSV格式的檔案
df = pd.read_csv("stocks\\2330.TW.csv", encoding="utf8")
df = df.dropna()

output_file("Ch16_4_3a.html")

data = ColumnDataSource(data={
        "close": df["Close"],
        "volume": df["Volume"]
        })

p = figure(title="台積電的收盤價與成交量", 
           plot_height=400, plot_width=700, 
           x_range=(min(df.Close), max(df.Close)),
           y_range=(min(df.Volume), max(df.Volume)))
p.diamond(x="close", y="volume", source=data)
p.xaxis.axis_label = "2017年收盤價"
p.yaxis.axis_label = "2017年成交量"

show(p)