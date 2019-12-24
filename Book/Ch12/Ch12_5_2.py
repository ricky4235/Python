import pandas as pd

df = pd.read_csv("products.csv", encoding="utf8")

df.columns = ["type", "name", "price"]
ordinals = ["A", "B", "C", "D", "E", "F"]
df.index = ordinals

# 呼叫 pivot_table() 方法
pivot_products = df.pivot_table(index='type',
                                columns='name',
                                values='price')
print(pivot_products)
pivot_products.to_html("Ch12_5_2.html")