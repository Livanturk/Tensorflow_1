import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
import seaborn as sbn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the data
df = pd.read_excel("Data/bisiklet_fiyatlari.xlsx")



# y = wx+b -> y = fiyat, x = bisiklet w = bias
# y -> label
y = df["Fiyat"].values # numpy array



# x -> feature
x = df[["BisikletOzellik1", "BisikletOzellik2"]].values # numpy array


# Seperating the data as test/train
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42) # 33% of the data is used for testing, 67% for training, random_state is the seed for the random number generator



# scale the data to make it easier for the model to learn
scaler = MinMaxScaler()
scaler.fit(x_train) # fit the scaler to the training data



x_train = scaler.transform(x_train) # scale the training data
x_test = scaler.transform(x_test) # scale the test data



# Create a model
model = Sequential()



# Add layers to the model
model.add(Dense(4, activation = "relu")) # 4 neurons, relu activation function
model.add(Dense(4, activation = "relu"))
model.add(Dense(4, activation = "relu"))
model.add(Dense(1)) # 1 neuron, no activation function



# Compile the model
model.compile(optimizer = "rmsprop", loss = "mse") # rmsprop optimizer, mean squared error loss function



# Fit the model
model.fit(x_train, y_train, epochs = 250) # train the model, epochs = number of iterations over the entire dataset



loss = model.history.history["loss"] # get the loss values
print(loss) # print the loss values



# Plot the loss values
sbn.lineplot(x = range(len(loss)), y = loss) # plot the loss values
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show() # show the plot



# Evaluate the model
training_loss = model.evaluate(x_train, y_train, verbose = 0) # evaluate the model on the training data and verbose = 0 means no output, verbose = 1 means output, this code will output the loss value
test_loss = model.evaluate(x_test, y_test, verbose = 0) # evaluate the model on the test data
print("Training Loss: ", training_loss)
print("Test Loss: ", test_loss)



# Predictions
testPredictions = model.predict(x_test) # make predictions on the test data
print(testPredictions) # print the predictions



# Comparing the predictions with the real values
predictionDf = pd.DataFrame(y_test, columns = ["Real Y"]) # create a dataframe with the real y values
testPredictions = pd.Series(testPredictions.reshape(330,)) # reshape the predictions to a series



# Concatenate the dataframes
predictionDf = pd.concat([predictionDf, testPredictions], axis = 1) # concatenate the dataframes along the columns axis = 1 (horizontally)
predictionDf.columns = ["Real Y", "Predicted Y"] # rename the columns
print(predictionDf) # print the dataframe



# Plot the real y values and the predicted y values
sbn.scatterplot(x = "Real Y", y = "Predicted Y", data = predictionDf) # scatter plot of the real y values and the predicted y values
plt.show() # show the plot



# Calculate the error metrics
mae = mean_absolute_error(predictionDf["Real Y"], predictionDf["Predicted Y"])
mse = mean_squared_error(predictionDf["Real Y"], predictionDf["Predicted Y"])
print("Mean Absolute Error: ", mae)
print("Mean Squared Error: ", mse)



# Describe the data
print(df.describe()) # describe the data



# Add a new column to the dataframe
newBisikletOzellikleri = [[1760, 1758]]
newBisikletOzellikleri = scaler.transform(newBisikletOzellikleri) # scale the new data  before making predictions

newPredictions =model.predict(newBisikletOzellikleri) # make predictions on the new data
print(newPredictions) # print the predictions