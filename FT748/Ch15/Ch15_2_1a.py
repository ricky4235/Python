from bokeh.plotting import figure, output_file, show

x = [0, 1, 2, 3, 4]
y = [5, 10, 15, 21, 31]

output_file("Ch15_2_1a.html")

p = figure()
p.vbar(x, top=y, color="blue", width=0.5)

show(p)