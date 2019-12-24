from bokeh.plotting import figure, output_file, show
from bokeh.plotting import ColumnDataSource
import pandas as pd

# 匯入CSV格式的檔案
df = pd.read_csv("stocks\\2330.TW.csv", encoding="utf8")
df = df.dropna()
df["Date"] = pd.to_datetime(df["Date"])

output_file("Ch16_4_3b.html")

data = ColumnDataSource(data={
        "date": df["Date"],
        "close": df["Close"]
        })

p = figure(title="台積電2017年的每日收盤價",
           plot_height=400, plot_width=700,
           x_axis_type="datetime",
           x_range=(min(df.Date), max(df.Date)),
           y_range=(min(df.Close), max(df.Close)))
p.line(x="date", y="close", source=data)
p.diamond(x="date", y="close", source=data)
p.xaxis.axis_label = "2017年"
p.yaxis.axis_label = "收盤價"

show(p)