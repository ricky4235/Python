import pandas as pd
import random

df = pd.DataFrame([random.sample(range(0,1000), 3),
                   random.sample(range(0,1000), 3)])
print(df)
df.to_html("Ch12_4_1c_01.html")
# 取得與更新整個DataFrame
print(df[df > 500])
df[df > 500].to_html("Ch12_4_1c_02.html")
df[df > 500] = df - 100
print(df)
df.to_html("Ch12_4_1c_03.html")