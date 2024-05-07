import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.layers import Dense, Activation, Dropout # Dropout is a technique where randomly selected neurons are ignored during training, this helps prevent overfitting
from keras.models import Sequential
from keras.callbacks import EarlyStopping # Early stopping is a method that allows you to specify an arbitrary large number of training epochs and stop training once the model performance stops improving on a hold out validation dataset (preventing overfitting)
from sklearn.metrics import classification_report, confusion_matrix



df = pd.read_excel("Data/maliciousornot.xlsx")
print(df)
print(df.describe().transpose())


""" correlation = df.corr()["Type"].sort_values().plot(kind= "bar")
print(correlation)

sbn.countplot(x = "Type", data = df) 
plt.show() """



# Drop the 'Type' column
y = df["Type"].values # numpy array
x = df.drop("Type", axis = 1).values # all columns except the 'Type' column



# Split the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.3, random_state = 15)



# Scale the data
scaler = MinMaxScaler() # scale the data to make it easier for the model to learn
x_train_scaled = scaler.fit_transform(x_train) # Scale and transform the training data
x_test_scaled = scaler.transform(x_test) # Only transform the test data using the same scaler as used for training
# x_train shape = (383, 30)



# Create a model
model = Sequential() # create a Sequential model
model.add(Dense(units = 30, activation = "relu")) # add a Dense layer with 30 neurons and relu activation function, 30 neurons because there are 30 features
model.add(Dropout(0,5)) # add a Dropout layer with 0.5 dropout rate, this means 50% of the neurons will be ignored during training
model.add(Dense(units = 15, activation = "relu"))
model.add(Dropout(0,5))
model.add(Dense(units = 15, activation = "relu"))
model.add(Dropout(0,5))
model.add(Dense(units = 1, activation = "sigmoid")) # add a Dense layer with 1 neuron and sigmoid activation function, this is the output layer, sigmoid activation function is used for binary classification problems



# Compile the model
model.compile(loss = "binary_crossentropy", optimizer = "adam") # binary_crossentropy loss function, adam optimizer is used for binary classification problems



# Early stopping
earlyStopping = EarlyStopping(monitor= "val_loss", mode = "min", verbose =1, patience = 25)



# Fit the model
model.fit(x = x_train, y = y_train, epochs = 700, validation_data = (x_test, y_test), verbose = 1, callbacks = [earlyStopping]) # 700 epochs, validation data is used to evaluate the model, verbose = 1 means show the progress bar, verbose = 0 means do not show the progress bar



# Model evalutation
modelLoss = pd.DataFrame(model.history.history) # get the loss values from the model
modelLoss.plot() # plot the loss values
plt.show() # show the plot


# Predictions
predictions = model.predict(x_test) # make predictions on the test data
predictions = np.argmax(predictions, axis=1) # convert probabilities to class predictions

# Classification report
report = classification_report(y_test, predictions) # classification report
print(report) # print the classification report

# Confusion matrix
confusionMatrix = confusion_matrix(y_test, predictions) # confusion matrix
print(confusionMatrix) # print the confusion matrix