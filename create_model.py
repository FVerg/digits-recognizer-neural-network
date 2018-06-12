# ------------------------------------------------------------------------------
# This script creates a keras model and saves it in Models folder
# It has been used for the creation of 10 neural networks, one for each digit
# in order to recognize whether or not an image (handwritten digit) represent
# a specific number.
# ------------------------------------------------------------------------------

from keras.models import Sequential
from keras.layers import Dense

import numpy as np

np.random.seed(7)

# ------------------------------------------------------------------------------
# Select the number you want to train the network to recognize
# ------------------------------------------------------------------------------
rec_number = 8

# ------------------------------------------------------------------------------
# Set the network parameters
# ------------------------------------------------------------------------------
INPUT_NEURONS = 784
HIDDEN_NEURONS = 10
OUTPUT_NEURONS = 1
NUM_EPOCHS = 30
BATCH_SIZE = 10

# ------------------------------------------------------------------------------
# Choose the input file, considering the number you want the network to recognize
# - training_x: ClassLabel = 1 for number x
# ------------------------------------------------------------------------------

dataset = np.loadtxt(r"mnist_datasets2\training_"+str(rec_number)+".csv", delimiter=',', skiprows=1)

X = dataset[:, 1:785]
Y = dataset[:, 785]

model = Sequential()

# Layer 1 - Input: 784 Neurons
model.add(Dense(INPUT_NEURONS, input_dim=INPUT_NEURONS, activation='relu'))
# Layer 2 - Hidden: 10 Neurons
model.add(Dense(HIDDEN_NEURONS, activation='sigmoid'))
# Layer 3 - Output: 1 Neuron (T/F)
model.add(Dense(OUTPUT_NEURONS, activation='sigmoid'))

# Compile model
# - binary_crossentropy = Error function
# - optimizer = Gradient descent algorithm
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, Y, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE)

# Evaluate the model
scores = model.evaluate(X, Y)  # Evaluating the model using the same dataset
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# Save the model on disk
# ------------------------------------------------------------------------------
# Change the output name depending on what number this NN has been trained on.
# ------------------------------------------------------------------------------
# model.save("Models\model_"+str(rec_number)+".h5")
