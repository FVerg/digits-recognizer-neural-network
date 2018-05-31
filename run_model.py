# -----------------------------------------------------------------------------
# In this script we are going to run the 10 different Neural Networks, and
# predict the classification of an input image, passed to all of them.
# The main idea is to run those neural networks in parallel, but I don't know
# whether my GPU is able to do that or not.
# -----------------------------------------------------------------------------

import numpy as np
from keras.models import load_model

# -----------------------------------------------------------------------------
# Change the input file according to the number you want to predict.
# Each of the 10 different NNs has a training and test set, that MUST not be
# confused with the others (Do not run NN0 with test_1.csv)
# -----------------------------------------------------------------------------
dataset = np.loadtxt(r"mnist_datasets2\test_0.csv", delimiter=',', skiprows=1)

# X is a ndarray containing all the test set images
X = dataset[:, 1:785]
# Y is a ndarray containing whether the
Y = dataset[:, 785]

model = load_model("Models\model_0.h5")

# Choose the images to feed the network in order to predict their represented value
predictions = model.predict(X[0:10])

# Some debug prints, can be ignored.
print("X[0] : ", X[0], " len(X[0]): ", len(X[0]))
print("Input data: ", len(X))
print("Predictions: ", len(predictions))

# round predictions
rounded = [round(x[0]) for x in predictions]
print(rounded)
