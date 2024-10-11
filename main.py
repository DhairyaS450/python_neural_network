import numpy as np
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork

input_vectors = np.array(
    [
        [3, 1.5],
        [2, 1],
        [4, 1.5],
        [3, 4],
        [3.5, 0.5],
        [2, 0.5],
        [5.5, 1],
        [1, 1],
    ]
)

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate = 0.001
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 100000)

plt.plot(training_error)
plt.xlabel('Iterations')
plt.ylabel('Error for all training instances')
plt.savefig('cumulative_error.png')