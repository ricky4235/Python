from bokeh.plotting import figure, output_file, show

x = [0, 1, 2, 3, 4]
y = [-1, -4.3, 15, 21, 31]

output_file("Ch15_1_2.html")

p = figure()
p.line(x, y, line_width=2)

show(p)