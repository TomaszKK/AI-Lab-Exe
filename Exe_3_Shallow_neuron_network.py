import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


class ShallowNeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size=2, learning_rate=0.01, activation_functions=None):
        self.learning_rate = learning_rate
        self.activation_functions = activation_functions or ['sigmoid'] * len(hidden_layers)
        self.layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(self.initialize_layer(prev_size, hidden_size))
            prev_size = hidden_size
        self.layers.append(self.initialize_layer(prev_size, output_size))

    def initialize_layer(self, input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        weights = np.random.uniform(-limit, limit, size=(input_size, output_size))
        biases = np.zeros(output_size)
        return {'weights': weights, 'biases': biases}

    def activate(self, x, activation_fn):
        if activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation_fn == 'heaviside':
            return np.where(x >= 0, 1, 0)
        elif activation_fn == 'relu':
            return np.maximum(0, x)
        elif activation_fn == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif activation_fn == 'tanh':
            return np.tanh(x)
        elif activation_fn == 'sin':
            return np.sin(x)
        elif activation_fn == 'sign':
            return np.sign(x)

    def activation_derivative(self, x, activation_fn):
        if activation_fn == 'sigmoid':
            return x * (1 - x)
        elif activation_fn == 'heaviside':
            return 1
        elif activation_fn == 'relu':
            return np.where(x > 0, 1, 0)
        elif activation_fn == 'leaky_relu':
            return np.where(x > 0, 1, 0.01)
        elif activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif activation_fn == 'sin':
            return np.cos(x)
        elif activation_fn == 'sign':
            return 1

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x, axis=0)


    def forward(self, inputs):
        activations = inputs
        for i, layer in enumerate(self.layers[:-1]):
            z = np.dot(activations, layer['weights']) + layer['biases']
            activations = self.activate(z, self.activation_functions[i])
            layer['activations'] = activations
            layer['z'] = z

        output_layer = self.layers[-1]
        z = np.dot(activations, output_layer['weights']) + output_layer['biases']
        activations = self.softmax(z)
        output_layer['activations'] = activations
        output_layer['z'] = z
        return activations


    def backward(self, inputs, expected_output):
        deltas = []
        output = self.layers[-1]['activations']
        error = expected_output - output
        delta = error * self.activation_derivative(output, 'sigmoid')
        deltas.append(delta)

        for i in reversed(range(len(self.layers) - 1)):
            layer = self.layers[i]
            next_layer = self.layers[i + 1]
            activation_fn = self.activation_functions[i]
            delta = np.dot(deltas[-1], next_layer['weights'].T) * self.activation_derivative(layer['activations'],
                                                                                             activation_fn)
            deltas.append(delta)

        deltas.reverse()
        activation = inputs
        for i in range(len(self.layers)):
            layer = self.layers[i]
            delta = deltas[i]
            layer['weights'] += self.learning_rate * np.outer(activation, delta)
            layer['biases'] += self.learning_rate * delta
            activation = layer['activations']

    def train(self, inputs, expected_output, epochs=1000):
        for epoch in range(epochs):
            for x, d in zip(inputs, expected_output):
                self.forward(x)
                self.backward(x, d)


def generate_class_data(num_modes, num_samples, class_label, activation_function):
    x_data = []
    y_data = []
    labels = []

    for mode in range(num_modes):
        mean_x = np.random.uniform(-1, 1)
        mean_y = np.random.uniform(-1, 1)
        variance_x = np.random.uniform(0.05, 0.5)
        variance_y = np.random.uniform(0.05, 0.5)
        x_mode = np.random.normal(mean_x, variance_x, num_samples)
        y_mode = np.random.normal(mean_y, variance_y, num_samples)
        x_data.extend(x_mode)
        y_data.extend(y_mode)
        if activation_function in ['tanh', 'sin', 'sign']:
            if class_label == 0:
                labels.extend([[-1, 1]] * num_samples)
            else:
                labels.extend([[1, -1]] * num_samples)
        else:
            if class_label == 0:
                labels.extend([[1, 0]] * num_samples)
            else:
                labels.extend([[0, 1]] * num_samples)

    return x_data, y_data, labels


def main():
    st.title("Shallow Neural Network")

    num_modes_blue = st.sidebar.text_input("Number of Modes Red", value=1)
    num_modes_red = st.sidebar.text_input("Number of Modes Blue", value=1)
    num_samples = st.sidebar.text_input("Number of Samples per Mode", value=50)
    num_hidden_layers = st.sidebar.slider("Number of Hidden Layers", 2, 5, 3)
    neurons_per_layer = st.sidebar.slider("Neurons per Hidden Layer", 1, 10, 5)
    activation_fun = ["sigmoid", "heaviside", "relu", "leaky_relu", "tanh", "sin", "sign"]
    activation_fn = []
    for i in range(num_hidden_layers):
        selected_activation = st.sidebar.selectbox(f"Activation for Layer {i + 1}", activation_fun, index=0)
        activation_fn.append(selected_activation)


    learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 1.0)
    epochs = st.sidebar.slider("Epochs", 1, 1000, 100)

    if st.sidebar.button("Generate & Train Neuron"):
        x_class0, y_class0, labels_class0 = generate_class_data(int(num_modes_blue), int(num_samples), class_label=0,                                                      activation_function='sigmoid')
        x_class1, y_class1, labels_class1 = generate_class_data(int(num_modes_red), int(num_samples), class_label=1,
                                                                activation_function='sigmoid')

        x_data = np.column_stack((x_class0 + x_class1, y_class0 + y_class1))
        labels = np.array(labels_class0 + labels_class1)

        hidden_layers = [neurons_per_layer] * num_hidden_layers
        nn = ShallowNeuralNetwork(input_size=2, hidden_layers=hidden_layers, output_size=2,
                                  learning_rate=learning_rate, activation_functions=activation_fn)
        nn.train(x_data, labels, epochs=epochs)

        xx, yy = np.meshgrid(np.linspace(min(x_class0 + x_class1), max(x_class0 + x_class1), 200),
                             np.linspace(min(y_class0 + y_class1), max(y_class0 + y_class1), 200))
        grid_points = np.stack([xx.flatten(), yy.flatten()], axis=1)
        zz = np.array([nn.forward(point)[1] for point in grid_points])
        zz = zz.reshape(xx.shape)

        plt.contourf(xx, yy, zz, levels=50, cmap="RdBu", alpha=0.6)
        # plt.contourf(xx, yy, zz, alpha=0.8, cmap=plt.cm.coolwarm)
        plt.scatter(x_class0, y_class0, color="red", label="Class 0")
        plt.scatter(x_class1, y_class1, color="blue", label="Class 1")
        plt.legend()
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")

        st.pyplot(plt)


if __name__ == "__main__":
    main()
