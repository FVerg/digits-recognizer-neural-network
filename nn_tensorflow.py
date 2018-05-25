import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  # This line allows Jupyter Notebook to print directly inline the images
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("/tmp/data", one_hot=True)

# Print one image of the dataset on the notebook
# plt.imshow(mnist.train.images[5].reshape(28, 28), cmap="Greys")

# Print the label assigned to the previous image
# mnist.train.labels[5]

# Network parameters:
n_input = 784               # One node for each pixel -> 28*28=784
hidden_layer_neurons = 300  # Number of hidden neurons
n_classes = 10              # Number of possible classes

# Training parameters:
learning_rate = 0.005
training_epochs = 30000
batch_size = 50

# Initializing tensorflow variables and models

x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# CREATING WEIGHT AND BIASES FOR THE NETWORK

# Weights from input to hidden layer
w1 = tf.Variable(tf.random_normal([n_input, hidden_layer_neurons]))
# Weights from hidden to output layer
w2 = tf.Variable(tf.random_normal([hidden_layer_neurons, n_classes]))

# Biases for hidden neurons
b1 = tf.Variable(tf.random_normal([hidden_layer_neurons]))
# Biases for output neurons
b2 = tf.Variable(tf.random_normal([n_classes]))

# MULTILAYER PERCEPTRON MODEL

# For each hidden neuron apply sigm(wx + b)
hidden_layer = tf.nn.sigmoid(tf.add(tf.matmul(x, w1), b1))

# For each output neuron apply sigm(wx + b) on the output of the prev. layer
output_layer = tf.nn.sigmoid(tf.add(tf.matmul(hidden_layer, w2), b2))

# Cost function and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output_layer, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Define the Test model and accuracy
correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y, 1))
correct_prediction = tf.cast(correct_prediction, "float")
accuracy = tf.reduce_mean(correct_prediction)

# TENSORFLOW SESSION
sess = tf.InteractiveSession()

# Initialize Variables
init = tf.global_variables_initializer()

# Start session
sess.run(init)

# Accuracies arrays to create a plot
train_accuracies = []
validation_accuracies = []
epoc_iteration = []

# Run the session, save accuracies
for epoch in range(training_epochs):
    batch_x, batch_y = mnist.train.next_batch(batch_size)

    if (epoch+1) < 100 or (epoch+1) % 100 == 0:
        train_ac = accuracy.eval({x: batch_x, y: batch_y})
        validation_ac = accuracy.eval({x: mnist.validation.images, y: mnist.validation.labels})

        epoc_iteration.append(epoch+1)
        train_accuracies.append(train_ac)
        validation_accuracies.append(validation_ac)

    sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})

# Plot the training and validation accuracies
# Create black canvas

fig = plt.figure(figsize=(10, 7))
axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes2 = fig.add_axes([0.36, 0.25, 0.53, 0.5])

# Plot full graph

axes1.plot(epoc_iteration, train_accuracies, '-b', label='Training')
axes1.plot(epoc_iteration, validation_accuracies, '-g', label='Validation')
axes1.legend()
axes1.set_xlabel('Epoch')
axes1.set_ylabel('Accuracy')
axes1.set_title('Training and Validation accuracy')

# Plot zoom in graph
plt.ylim(max=1.001, ymin=0.95)
axes2.plot(epoc_iteration[198:], train_accuracies[198:], '-b', label='Training')
axes2.plot(epoc_iteration[198:], validation_accuracies[198:], '-g', label='Validation')
axes2.set_title('Zoom in')

# Print final accuracies
print("Validation accuracy: ", accuracy.eval(
    {x: mnist.validation.images, y: mnist.validation.labels}))
print("Test accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
