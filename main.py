import pandas as pd

filepath = 'C:/Users/7119001/Downloads/202501_CombinedData.csv'
df = pd.read_csv(filepath)

print(df.info())