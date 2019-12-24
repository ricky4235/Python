import pandas as pd

products = {"分類": ["居家","居家","娛樂","娛樂","科技","科技"],
            "商店": ["家樂福","大潤發","家樂福","全聯超","大潤發","家樂福"],
            "價格": [11.42,23.50,19.99,15.95,55.75,111.55]}

df = pd.DataFrame(products, 
                  columns = ["分類", "商店", "價格"]) 

df.to_csv("products.csv", index=False, encoding="utf8")
df.to_json("products.json")