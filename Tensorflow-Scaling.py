# Scale the data to make it easier for the model to learn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df = pd.read_excel("Data/bisiklet_fiyatlari.xlsx")

# y = wx+b -> y = fiyat, x = bisiklet w = bias

# y -> label
y = df["Fiyat"].values # numpy array

# x -> feature
x = df[["BisikletOzellik1", "BisikletOzellik2"]].values # numpy array

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42) # 33% of the data is used for testing, 67% for training, random_state is the seed for the random number generator

print(x_train) # before scaling

scaler = MinMaxScaler()
scaler.fit(x_train) # fit the scaler to the training data

x_train = scaler.transform(x_train) # scale the training data
x_test = scaler.transform(x_test) # scale the test data

print(x_train) # after scaling