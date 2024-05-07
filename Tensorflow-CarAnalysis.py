import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

df = pd.read_excel("Data/merc.xlsx")

# print(df)
#print(df.describe().transpose()) # print the data in a more readable format


nullValues = df.isnull().sum() # check for missing data
""" print(nullValues) """

#plt.figure(figsize = (7,5))
#sbn.distplot(df["price"]) # show the distribution of the price
#plt.show()



""" corr = df.corr(numeric_only=True)["price"].sort_values() # correlation matrix for numeric columns only, aim is to see the correlation between the price and other columns
print(corr) """
#sbn.scatterplot(x = "mileage", y = "price", data = df) # show the relationship between mileage and price



#sortedValues = df.sort_values("price", ascending = False).head(20) # sort the data by price in descending order
#print(sortedValues)



""" onePercentOfData = len(df) * 0.01 # 1% of the data
print(onePercentOfData) # 131.19

dfWithoutOutliers = df.sort_values("price", ascending = False).iloc[int(onePercentOfData):] # remove the top 1% of the data
describe = dfWithoutOutliers.describe().transpose() # describe the data in a more readable format
print(describe) """




#plt.figure(figsize = (7,5)) # create a figure, 7 inches wide, 5 inches tall
#sbn.distplot(dfWithoutOutliers["price"]) # show the distribution of the price
#plt.show()


""" # Group by the 'year' column and calculate the mean price excluding attributes containing string values
df['year'] = pd.to_numeric(df['year'], errors='coerce')
# Drop the 'transmission' column as it contains string values
df.drop('transmission', axis=1, inplace=True)
# Group by the 'year' column and calculate the mean price
groupedByYear = df.groupby("year").mean()["price"] # group by the year column and calculate the mean price
print(groupedByYear) """

# Drop the 'transmission' column as it contains string values
dfWithoutTransmission = df.drop("transmission", axis = 1) # drop the 'transmission' column

y = dfWithoutTransmission["price"].values
x = dfWithoutTransmission.drop("price", axis = 1).values # axis = 1 means drop the column, axis = 0 means drop the row

print(x)
print(y)