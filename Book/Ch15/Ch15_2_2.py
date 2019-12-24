from bokeh.plotting import figure, output_file, show

output_file("Ch15_2_2.html")

p = figure()
p.cross(1, 2, size=15)
p.x(2,2, size=15)
p.diamond(3,2, size=15)
p.diamond_cross(4, 2, size=15)
p.circle(5,2, size=15)
p.circle_x(6,2, size=15)
p.triangle(7,2, size=15)
p.inverted_triangle(8,2, size=15)
p.square(9,2, size=15)
p.asterisk(10,2,size=15)

show(p)