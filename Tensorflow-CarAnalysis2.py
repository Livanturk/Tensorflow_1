import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential # Sequential model is a linear stack of layers, we create a Sequential model and add layers to it
from keras.layers import Dense # Dense layer is a fully connected layer, we add Dense layers to the model
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Read the data
df = pd.read_excel("Data/merc.xlsx")


# Drop the 'transmission' column as it contains string values
dfWithoutTransmission = df.drop("transmission", axis = 1) #axis = 1 means drop the column, axis = 0 means drop the row



# y = wx+b -> y = price, x = car, w = bias
y = dfWithoutTransmission["price"].values
x = dfWithoutTransmission.drop("price", axis = 1) # drop the price column



# Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 10) # 30% of the data is used for testing, 70% for training, random_state is the seed for the random number generator



# Check if the data is split correctly
print(len(x_train))
print(len(x_test))



# Scale the data
scaler = MinMaxScaler() # scale the data to make it easier for the model to learn
x_train = scaler.fit_transform(x_train) # fit the scaler to the training data and scale the training data
x_test = scaler.fit_transform(x_test) # fit the scaler to the test data and scale the test data



# Create a model
model = Sequential() # create a Sequential model
model.add(Dense(12, activation = "relu")) # add a Dense layer with 12 neurons and relu activation function
model.add(Dense(12, activation = "relu")) # add a Dense layer with 12 neurons and relu activation function
model.add(Dense(12, activation = "relu")) # add a Dense layer with 12 neurons and relu activation function
model.add(Dense(12, activation = "relu")) # add a Dense layer with 12 neurons and relu activation function
model.add(Dense(1)) # add a Dense layer with 1 neuron and no activation function, this is the output layer



# Compile the model
model.compile(optimizer = "adam", loss = "mse") # adam optimizer, mean squared error loss function



# Fit the model
model.fit(x = x_train, y = y_train ,validation_data = (x_test, y_test), batch_size = 250 ,epochs = 300)



# Loss values
lossData = pd.DataFrame(model.history.history)
print(lossData.head())
lossData.plot() # plot the loss values
plt.show() # show the plot


# Predictions
predictions = model.predict(x_test) # make predictions
print(mean_absolute_error(y_test, predictions)) # mean absolute error
print(mean_squared_error(y_test, predictions)) # mean squared error 




# Show the actual price vs the predicted price using a scatter plot
plt.scatter(y_test, predictions) # scatter plot of the actual price vs the predicted price
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.plot(y_test, y_test, "r-*") # plot the line y = x
plt.show()



# Predict the price of a new car
newCar = dfWithoutTransmission.drop("price", axis = 1).iloc[2] # drop the price column and get the third row
newCar = scaler.transform(newCar.values.reshape(-1,5)) # scale the data and reshape it as the model expects a 2D array, -1 means unknown number of rows, 5 means 5 columns
predictedPrice = model.predict(newCar) # predict the price of the new car
print(predictedPrice) # print the predicted price