from bokeh.plotting import figure, output_file, show

x_region = [[2,1,2],[3,2,3],[3,4,5,4]]
y_region = [[2,4,6],[4,6,7],[3,4,7,8]]

output_file("Ch15_2_1d.html")

p = figure()
p.patches(x_region, y_region, fill_color=["yellow","red","green"],
          line_color="black")

show(p)