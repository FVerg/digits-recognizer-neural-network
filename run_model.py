# -----------------------------------------------------------------------------
# In this script we are going to run the 10 different Neural Networks, and
# predict the classification of an input image, passed to all of them.
# The main idea is to run those neural networks in parallel, but I don't know
# whether my GPU is able to do that or not.
# -----------------------------------------------------------------------------

import numpy as np
from keras.models import load_model
from keras.utils import plot_model

# -----------------------------------------------------------------------------
# Change the input file according to the number you want to predict.
# Each of the 10 different NNs has a training and test set, that MUST not be
# confused with the others (Do not run NN0 with test_1.csv)
# -----------------------------------------------------------------------------
dataset = np.loadtxt(r"mnist_datasets2\test_0.csv", delimiter=',', skiprows=1)

# X is a ndarray containing all the test set images
X = dataset[:, 1:785]
# Y is a ndarray containing whether the image is a number or not
Y = dataset[:, 785]

# -----------------------------------------------------------------------------
# Dictionary containing ndarray for labels of each number in test set
# labels[0]: For each of the 10k test images: 1 if it is a 0, 0 if it is not
# labels[1]: For each of the 10k test images: 1 if it is a 1, 0 if it is not
# [...]
# -----------------------------------------------------------------------------
labels = {0: Y, 1: None, 2: None, 3: None, 4: None,
          5: None, 6: None, 7: None, 8: None, 9: None}

#print("L'elemento ", 0, "ha ", len(labels[0]), " labels")
for i in range(1, 10):
    dataset = np.loadtxt(r"mnist_datasets2\test_"+str(i)+".csv", delimiter=',', skiprows=1)
    labels[i] = dataset[:, 785]
    #print("L'elemento ", i, "ha ", len(labels[i]), " labels")


# -----------------------------------------------------------------------------
# Change the input model according to the network you want to load
# -----------------------------------------------------------------------------
models = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None, 6: None, 7: None, 8: None, 9: None}
for i in range(0, 10):
    models[i] = load_model("Models\model_"+str(i)+".h5")
#model = load_model("Models\model_0.h5")

# Save to png the structure of the model
# plot_model(model, to_file='model_0.png', show_shapes=True)

# Choose the images to feed the network in order to predict their represented value
predictions = {0: None, 1: None, 2: None, 3: None,
               4: None, 5: None, 6: None, 7: None, 8: None, 9: None}
for i in range(0, 10):
    predictions[i] = models[i].predict(X)

# Some debug prints, can be ignored.
print("X[0] : ", X[0], " len(X[0]): ", len(X[0]))


# round predictions

rounded_predictions = {0: None, 1: None, 2: None, 3: None,
                       4: None, 5: None, 6: None, 7: None, 8: None, 9: None}

for i in range(0, 10):
    rounded_predictions[i] = [round(x[0]) for x in predictions[i]]

predicted_labels = np.zeros(10000, dtype=int)

# Retrieving outputs from the neural networks
for i in range(0, 10):
    for j in range(0, 10000):
        if (rounded_predictions[i][j] == 1.0 and predicted_labels[j] == 0):
            predicted_labels[j] = i

base = np.loadtxt(r"mnist_datasets2\test_base.csv", delimiter=',', skiprows=1)
correct_labels = base[:, 785]

# Calculate accuracy
errors = 0
for i, j in zip(predicted_labels, correct_labels):
    print("Predicted: ", i, " - Correct: ", int(j))
    if i != int(j):
        errors = errors+1
print(errors, " mistaken prediction over a test set of 10000 elements")
print("Accuracy = ", ((10000-errors)/10000)*100, "%")
# print(rounded_predictions)
