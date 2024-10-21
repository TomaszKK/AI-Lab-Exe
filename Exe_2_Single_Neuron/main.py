import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

#check bias
# Activation functions
def heaviside(x):
    return np.where(x >= 0, 1, 0)


def sigmoid(x, beta=1):
    return 1 / (1 + np.exp(-beta * x))


def tanh(x):
    return np.tanh(x)


def sin(x):
    return np.sin(x)


def sign(x):
    return np.sign(x)


def relu(x):
    return np.maximum(0, x)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


# Derivative functions
def sigmoid_derivative(x, beta=1):
    sig = sigmoid(x, beta)
    return sig * (1 - sig)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


def sin_derivative(x):
    return np.cos(x)


def sign_derivative(x):
    return 1


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


# Single neuron model
class SingleNeuron:
    def __init__(self, activation='heaviside', learning_rate=0.1, variable_lr=False):
        self.activation_name = activation
        self.learning_rate = learning_rate
        self.variable_lr = variable_lr
        self.weights = None

    def activation(self, x):
        if self.activation_name == 'heaviside':
            return heaviside(x)
        elif self.activation_name == 'sigmoid':
            return sigmoid(x)
        elif self.activation_name == 'tanh':
            return tanh(x)
        elif self.activation_name == 'sin':
            return sin(x)
        elif self.activation_name == 'sign':
            return sign(x)
        elif self.activation_name == 'relu':
            return relu(x)
        elif self.activation_name == 'leaky_relu':
            return leaky_relu(x)

    def activation_derivative(self, x):
        if self.activation_name == 'heaviside':
            return 1  # Assumed to be 1
        elif self.activation_name == 'sigmoid':
            return sigmoid_derivative(x)
        elif self.activation_name == 'tanh':
            return tanh_derivative(x)
        elif self.activation_name == 'sin':
            return sin_derivative(x)
        elif self.activation_name == 'sign':
            return sign_derivative(x)
        elif self.activation_name == 'relu':
            return relu_derivative(x)
        elif self.activation_name == 'leaky_relu':
            return leaky_relu_derivative(x)

    def predict(self, X):
        weighted_sum = np.dot(X, self.weights)
        return self.activation(weighted_sum)

    def fit(self, X, y, epochs=100, eta_min=0.01, eta_max=1.0):
        # Initialize weights
        self.weights = np.random.rand(X.shape[1])

        for epoch in range(epochs):
            if self.variable_lr:
                # Update learning rate dynamically using cosine annealing
                learning_rate = eta_min + (eta_max - eta_min) * (1 + np.cos(np.pi * epoch / epochs)) / 2
            else:
                learning_rate = self.learning_rate

            for i in range(X.shape[0]):
                xi = X[i]
                yi = y[i]
                weighted_sum = np.dot(xi, self.weights)
                y_pred = self.activation(weighted_sum)

                # Update weights based on the learning rule
                error = yi - y_pred
                self.weights += learning_rate * error * xi * self.activation_derivative(weighted_sum)


# Generate data (as given previously)
def generate_class_data(num_modes, num_samples):
    x_data, y_data = [], []
    for _ in range(num_modes):
        mean_x = np.random.uniform(-1, 1)
        mean_y = np.random.uniform(-1, 1)
        std_var_x = np.random.uniform(0.05, 0.1)
        std_var_y = np.random.uniform(0.05, 0.1)
        x_data.extend(np.random.normal(mean_x, std_var_x, num_samples))
        y_data.extend(np.random.normal(mean_y, std_var_y, num_samples))
    return np.array(x_data), np.array(y_data)


def main():
    st.title("Single Neuron with Decision Boundary")

    num_modes_blue = st.sidebar.text_input("Number of modes blue", value=1)
    num_modes_red = st.sidebar.text_input("Number of modes red", value=1)
    num_samples = st.sidebar.text_input("Number of samples per mode", value=50)
    activation_fn = st.sidebar.selectbox("Activation Function", ['heaviside', 'sigmoid', 'tanh', 'sin', 'sign', 'relu', 'leaky_relu'])
    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
    epochs = st.sidebar.slider("Epochs", 1, 1000, 100)
    use_variable_lr = st.sidebar.checkbox("Use Variable Learning Rate", value=False)
    eta_min = st.sidebar.slider("Min Learning Rate", 0.01, 1.0, 0.01)
    eta_max = st.sidebar.slider("Max Learning Rate", 0.01, 1.0, 1.0)

    if st.sidebar.button("Generate & Train Neuron"):
        blue_x, blue_y = generate_class_data(int(num_modes_blue), int(num_samples))
        red_x, red_y = generate_class_data(int(num_modes_red), int(num_samples))

        red_labels = np.zeros_like(red_x)  # Label red as 0
        blue_labels = np.ones_like(blue_x)  # Label blue as 1

        # Combine data
        X = np.vstack([np.column_stack([red_x, red_y]), np.column_stack([blue_x, blue_y])])
        y = np.concatenate([red_labels, blue_labels])

        # Train the neuron
        neuron = SingleNeuron(activation=activation_fn, learning_rate=learning_rate, variable_lr=use_variable_lr)
        neuron.fit(X, y, epochs=epochs, eta_min=eta_min, eta_max=eta_max)

        # Create a meshgrid for plotting decision boundary
        x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
        y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        grid = np.c_[xx.ravel(), yy.ravel()]

        # Predict over the grid
        Z = neuron.predict(grid)
        Z = Z.reshape(xx.shape)

        # Plot decision boundary
        plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        plt.scatter(red_x, red_y, color="blue", label="Red Class")
        plt.scatter(blue_x, blue_y, color="red", label="Blue Class")
        plt.legend()
        plt.xlabel("X")
        plt.ylabel("Y")

        # Display the plot in Streamlit
        st.pyplot(plt)


if __name__ == "__main__":
    main()
