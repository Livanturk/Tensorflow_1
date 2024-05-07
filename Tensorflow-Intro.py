import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

df = pd.read_excel("bisiklet_fiyatlari.xlsx")
print(df) # print all data
print(df.head()) # print first 5 data
sbn.pairplot(df) # show all data in graph
plt.show() # show graph