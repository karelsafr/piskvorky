import numpy as np
from scipy.stats import entropy
import os


class tym_girls:
    def __init__(self):
        self.v_engine = "1.0.0"
        self.board = None

        # Path to the parameter file
        current_dir = os.path.dirname(__file__)
        self.param_file = os.path.join(current_dir, "temp_engines", "winner_engine.npz")

        # Check if the parameter file exists and load it
        if os.path.exists(self.param_file):
            #print(f"Loading parameters from {self.param_file}")
            self.load_params(self.param_file)
        else:
            print(f"No parameter file found at {self.param_file}. Initializing randomly.")
            self.initialize_parameters()



    def initialize_parameters(self):
        # Input size for a 5x5 board is 25
        input_size = 25
        hidden_layer_1 = 64  # Neurons in the first hidden layer
        hidden_layer_2 = 32  # Neurons in the second hidden layer
        output_size = 1      # Single evaluation score

        # Weights and biases for two hidden layers and one output layer
        self.W1 = np.random.randn(hidden_layer_1, input_size) * 0.01
        self.b1 = np.zeros((hidden_layer_1, 1))
        self.W2 = np.random.randn(hidden_layer_2, hidden_layer_1) * 0.01
        self.b2 = np.zeros((hidden_layer_2, 1))
        self.W3 = np.random.randn(output_size, hidden_layer_2) * 0.01
        self.b3 = np.zeros((output_size, 1))

        # Store parameters in a dictionary
        self.parameters = {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
            'W3': self.W3,
            'b3': self.b3
        }

    def load_params(self, file):
        """
        Load pre-trained parameters from a file.
        """
        saved_params = np.load(file)
        self.parameters = {key: saved_params[key] for key in saved_params.files}

        # Set the loaded parameters to the model
        self.W1, self.b1 = self.parameters['W1'], self.parameters['b1']
        self.W2, self.b2 = self.parameters['W2'], self.parameters['b2']
        self.W3, self.b3 = self.parameters['W3'], self.parameters['b3']

        print("Parameters loaded successfully.")

    def relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-Z))

    def forward_pass(self, X):
        """
        Perform a forward pass through the neural network.
        X: Input board state flattened as a column vector.
        Returns the evaluation score.
        """
        # Input to first hidden layer
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)

        # First hidden to second hidden layer
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.relu(Z2)

        # Second hidden to output layer
        Z3 = np.dot(self.W3, A2) + self.b3
        A3 = self.sigmoid(Z3)

        return A3

    def evaluation(self):
        """
        Evaluate the board using the neural network.
        """
        # Flatten the board into a column vector
        X = self.board.reshape(-1, 1)

        # Forward pass through the network
        A3 = self.forward_pass(X)

        # Scale the output to the range [-10, 10]
        return (A3[0, 0] - 0.5) * 20

    def evaluate_board(self, board):
        """
        Main method to evaluate a given board.
        """
        self.board = board.copy()
        return self.evaluation()

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        """
        Mutate the neural network parameters intelligently.
        mutation_rate: Probability of mutation for each parameter.
        mutation_scale: Strength of mutation.
        """
        for param_name in self.parameters:
            mutation_mask = np.random.random(self.parameters[param_name].shape) < mutation_rate
            mutations = np.random.randn(*self.parameters[param_name].shape) * mutation_scale
            self.parameters[param_name] += mutations * mutation_mask

        # Update references to parameters
        self.W1, self.b1 = self.parameters['W1'], self.parameters['b1']
        self.W2, self.b2 = self.parameters['W2'], self.parameters['b2']
        self.W3, self.b3 = self.parameters['W3'], self.parameters['b3']

    def incorporate_symmetry(self, board):
        """
        Process the board with symmetry to normalize its representation.
        This helps the neural network learn equivalent positions efficiently.
        """
        # For example, return the board or its rotation/reflection with the "smallest" representation.
        rotations = [np.rot90(board, k) for k in range(4)]
        reflections = [np.fliplr(r) for r in rotations]
        all_transformations = rotations + reflections
        return min(all_transformations, key=lambda b: b.tobytes())

    def evaluate_symmetrical_board(self, board):
        """
        Evaluate the board while considering symmetrical states.
        """
        normalized_board = self.incorporate_symmetry(board)
        return self.evaluate_board(normalized_board)

    def train(self, training_data, labels, epochs=10, learning_rate=0.01):
        """
        Train the neural network using gradient descent.
        training_data: Array of board states (flattened).
        labels: Array of evaluation scores for the board states.
        epochs: Number of training iterations.
        learning_rate: Step size for weight updates.
        """
        for epoch in range(epochs):
            for X, y in zip(training_data, labels):
                X = X.reshape(-1, 1)  # Ensure column vector
                y = np.array([[y]])  # Ensure column vector

                # Forward pass
                Z1 = np.dot(self.W1, X) + self.b1
                A1 = self.relu(Z1)
                Z2 = np.dot(self.W2, A1) + self.b2
                A2 = self.relu(Z2)
                Z3 = np.dot(self.W3, A2) + self.b3
                A3 = self.sigmoid(Z3)

                # Compute loss (Mean Squared Error)
                loss = (A3 - y) ** 2

                # Backward pass
                dA3 = 2 * (A3 - y)
                dZ3 = dA3 * A3 * (1 - A3)  # Sigmoid derivative
                dW3 = np.dot(dZ3, A2.T)
                db3 = dZ3

                dA2 = np.dot(self.W3.T, dZ3)
                dZ2 = dA2 * (A2 > 0)  # ReLU derivative
                dW2 = np.dot(dZ2, A1.T)
                db2 = dZ2

                dA1 = np.dot(self.W2.T, dZ2)
                dZ1 = dA1 * (A1 > 0)  # ReLU derivative
                dW1 = np.dot(dZ1, X.T)
                db1 = dZ1

                # Update weights and biases
                self.W3 -= learning_rate * dW3
                self.b3 -= learning_rate * db3
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1

    def load_params(self, file=""):
        """
        Load parameters from a file.
        """
        self.set_parameters(np.load(file))

    def get_parameters(self):
        """
        Get a copy of all parameters.
        """
        return self.parameters.copy()

    def set_parameters(self, parameters):
        """
        Set parameters for the neural network.
        """
        if isinstance(parameters, np.lib.npyio.NpzFile):
            # Převedení NpzFile na slovník
            self.parameters = {key: parameters[key] for key in parameters.files}
        else:
            # Pokud už je to slovník, uděláme kopii
            self.parameters = parameters.copy()
        self.W1, self.b1 = self.parameters['W1'], self.parameters['b1']
        self.W2, self.b2 = self.parameters['W2'], self.parameters['b2']
        self.W3, self.b3 = self.parameters['W3'], self.parameters['b3']