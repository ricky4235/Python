import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("NBA_salary_rankings_2018.csv")
df = df.sort_values("pos")  # 使用位置排序
# 找出位置清單
col = df.drop_duplicates(["pos"])
# 建立各位置薪水的巢狀清單
data = []
for pos in col["pos"].values:
    d = df[(df.pos == pos)]
    data.append(d["salary"].values)
# 繪製箱形圖
plt.boxplot(data)
plt.xticks(range(1,6), col["pos"], rotation=25)
plt.title("Box Plot of NBA Salary") 
plt.show()