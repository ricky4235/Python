from bokeh.plotting import figure, output_file, show

x = [0, 1, 2, 3, 4]
y = [5, 10, 15, 21, 31]

output_file("Ch15_2_2a.html")

p = figure(title="Bakeh的折線圖", 
           title_location="above",
           x_axis_label="X軸",
           y_axis_label="Y軸")
p.line(x, y, line_width=2)
p.cross(x, y, size=10)

show(p)