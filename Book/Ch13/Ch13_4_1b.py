import matplotlib.pyplot as plt
import math

x = [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,
     5.5,6,6.5,7,7.5,8,8.5,9,9.5,10]

plt.subplot(231)
plt.plot(x, [math.sin(v) for v in x])
plt.subplot(232)
plt.plot(x, [math.cos(v) for v in x])
plt.subplot(233)
plt.plot(x, [math.tan(v) for v in x])
plt.subplot(234)
plt.plot(x, [math.sinh(v) for v in x])
plt.subplot(235)
plt.plot(x, [math.cosh(v) for v in x])
plt.subplot(236)
plt.plot(x, [math.tanh(v) for v in x])
plt.show()

