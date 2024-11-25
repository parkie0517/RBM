import numpy as np
from tensorflow.keras.datasets import mnist

# Load and preprocess MNIST data
def load_data():
    (x_train, _), (_, _) = mnist.load_data()
    x_train = (x_train / 255.0).reshape(-1, 784)  # Normalize and flatten
    return x_train

# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# RBM Class
class RBM:
    def __init__(self, visible_size, hidden_size):
        self.m = visible_size
        self.n = hidden_size
        self.W = np.random.normal(0, 0.01, (self.m, self.n))  # Weight matrix
        self.a = np.zeros(self.m)  # Visible bias
        self.b = np.zeros(self.n)  # Hidden bias

    def sample_prob(self, probs):
        return (probs > np.random.rand(*probs.shape)).astype(np.float32)

    def gibbs_sampling(self, v):
        # Positive phase
        h_prob = sigmoid(np.dot(v, self.W) + self.b)  # Hidden activations
        h_sample = self.sample_prob(h_prob)

        # Negative phase
        v_prob = sigmoid(np.dot(h_sample, self.W.T) + self.a)  # Reconstruct visible
        v_sample = self.sample_prob(v_prob)

        h_prob_neg = sigmoid(np.dot(v_sample, self.W) + self.b)  # Recalculate hidden
        return v_prob, h_prob, v_sample, h_prob_neg

    def train(self, data, batch_size=100, learning_rate=0.1, epochs=10):
        for epoch in range(epochs):
            np.random.shuffle(data)
            mini_batches = [data[k:k + batch_size] for k in range(0, len(data), batch_size)]

            for batch in mini_batches:
                v0 = batch

                # Gibbs Sampling
                v1, h0, v1_sample, h1 = self.gibbs_sampling(v0)

                # Update weights and biases
                dW = np.dot(v0.T, h0) - np.dot(v1_sample.T, h1)
                da = np.mean(v0 - v1_sample, axis=0)
                db = np.mean(h0 - h1, axis=0)

                # Gradient update
                self.W += learning_rate * dW / batch_size
                self.a += learning_rate * da
                self.b += learning_rate * db

            print(f"Epoch {epoch + 1}/{epochs} completed.")

    def reconstruct(self, v):
        h = sigmoid(np.dot(v, self.W) + self.b)
        v_reconstructed = sigmoid(np.dot(h, self.W.T) + self.a)
        return v_reconstructed

# Load data
data = load_data()

# Initialize and train RBM
visible_neurons = 784
hidden_neurons = 128
rbm = RBM(visible_size=visible_neurons, hidden_size=hidden_neurons)

# Train RBM
rbm.train(data, batch_size=100, learning_rate=0.1, epochs=1)

# Test reconstruction
sample = data[:10]
reconstructed = rbm.reconstruct(sample)

# Visualize original and reconstructed images
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
for i in range(10):
    # Original
    plt.subplot(2, 10, i + 1)
    plt.imshow(sample[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

    # Reconstructed
    plt.subplot(2, 10, i + 11)
    plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.show()