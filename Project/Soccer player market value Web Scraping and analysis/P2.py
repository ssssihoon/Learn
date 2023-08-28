import pandas as pd

df = pd.read_csv("transfermakt.csv")


g = df.groupby('nation')
print(g.sum())