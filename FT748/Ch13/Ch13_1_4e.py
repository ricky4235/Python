import matplotlib.pyplot as plt

days = range(0, 22, 3)
celsius1 = [25.6, 23.2, 18.5, 28.3, 26.5, 30.5, 32.6, 33.1]
celsius2 = [15.4, 13.1, 21.6, 18.1, 16.4, 20.5, 23.1, 13.2]

print(plt.style.available)

plt.style.use("ggplot")

plt.plot(days, celsius1, "r-o", label="Home")
plt.plot(days, celsius2, "g--", label="Office")
plt.legend()
plt.xlabel("Day")
plt.ylabel("Celsius")
plt.title("Home and Office Temperatures")
plt.show()

