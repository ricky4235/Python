from bokeh.plotting import figure, output_file, show

x = [0, 1, 2, 3, 4]
y = [5, 10, 15, 21, 31]

output_file("Ch15_2_1.html")

p = figure()
p.line(x, y, line_width=2)
p.cross(x, y, size=10)

show(p)