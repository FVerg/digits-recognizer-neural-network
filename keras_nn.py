from keras.models import Sequential
from keras.layers import Dense

import numpy as np

np.random.seed(7)

INPUT_NEURONS = 784
HIDDEN_NEURONS = 10
OUTPUT_NEURONS = 1
NUM_EPOCHS = 30
BATCH_SIZE = 10
# Load data for Class 0 images (Skip first line since it contains the header)
dataset = np.loadtxt("mnist_datasets\mnist_c0.csv", delimiter=',', skiprows=1)

X = dataset[:, 1:785]
Y = dataset[:, 785]

model = Sequential()

# Layer 1 - Input: 784 Neurons
model.add(Dense(INPUT_NEURONS, input_dim=INPUT_NEURONS, activation='relu'))
# Layer 2 - Hidden: 30 Neurons
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
model.save("Models\model_0_recognizer.hd5")
