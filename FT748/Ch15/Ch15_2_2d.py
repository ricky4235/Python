from bokeh.plotting import figure, output_file, show

x = [0, 1, 2, 3, 4]
y = [5, 10, 15, 21, 31]

output_file("Ch15_2_2d.html")

p = figure(title="Bakeh的折線圖", 
           title_location="above")
p.title.text_color = "red"
p.title.text_font_style = "bold"
p.xaxis.axis_label = "X軸"
p.xaxis.axis_label_text_color = "green"
p.yaxis.axis_label = "Y軸"
p.yaxis.axis_label_text_color = "blue"

p.line(x, y, line_width=2)
p.cross(x, y, size=10)

show(p)