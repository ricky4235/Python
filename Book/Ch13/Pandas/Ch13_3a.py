import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("NBA_salary_rankings_2018.csv")
# 繪出直方圖
num_bins = 15
df["salary"].plot.hist(bins=num_bins)
plt.ylabel("Frequency")
plt.xlabel("Salary")
plt.title("Histogram of NBA Top 100 Salary") 

