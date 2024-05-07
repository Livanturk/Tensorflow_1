# Seperating the data as test/train
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_excel("Data/bisiklet_fiyatlari.xlsx")

# y = wx+b -> y = fiyat, x = bisiklet w = bias

# y -> label
y = df["Fiyat"].values # numpy array

# x -> feature
x = df[["BisikletOzellik1", "BisikletOzellik2"]].values # numpy array

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42) # 33% of the data is used for testing, 67% for training, random_state is the seed for the random number generator

print("x_train shape: ", x_train.shape) # 670, 2 -> 670 rows, 2 columns (features)
print("x_test shape: ", x_test.shape) # 330, 2 -> 330 rows, 2 columns (features)
print("y_train shape:", y_train.shape) # 670, -> 670 rows, 1 column (label)
print("y_test shape:", y_test.shape) # 330, -> 330 rows, 1 column (label)
