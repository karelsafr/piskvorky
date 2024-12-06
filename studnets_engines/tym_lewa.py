import numpy as np


class LewaEngine:
    def __init__(self):
        self.v_engine = "0.1.0"
        self.board = None
        self.weights = None
        self.initialize_parameters()

    def initialize_parameters(self):
        # For a 5x5 board, we have 25 inputs
        input_size = 25

        # Initialize weights and biases for a simple two-layer neural network
        self.weights = {
            'W1': np.random.randn(32, input_size) * 0.01,  # First layer weights
            'b1': np.zeros((32, 1)),  # First layer biases
            'W2': np.random.randn(16, 32) * 0.01,  # Second layer weights
            'b2': np.zeros((16, 1)),  # Second layer biases
            'W3': np.random.randn(1, 16) * 0.01,  # Output layer weights
            'b3': np.zeros((1, 1))  # Output layer bias
        }

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        """
        Mutate the parameters of the neural network
        mutation_rate: Probability of mutating each parameter
        mutation_scale: Magnitude of the mutation
        """
        for key in self.weights:
            mutation_mask = np.random.random(self.weights[key].shape) < mutation_rate
            mutations = np.random.randn(*self.weights[key].shape) * mutation_scale
            self.weights[key] += mutation_mask * mutations

    def relu(self, Z):
        """ReLU activation function"""
        return np.maximum(0, Z)

    def tanh(self, Z):
        """Tanh activation function"""
        return np.tanh(Z)

    def evaluation(self):
        """
        Evaluate the board using the neural network
        """
        if self.board is None:
            raise ValueError("Board state is not set")

        # Flatten the board into a vector for the neural network input
        X = self.board.reshape(-1, 1)

        # Forward propagation
        Z1 = np.dot(self.weights['W1'], X) + self.weights['b1']
        A1 = self.relu(Z1)
        Z2 = np.dot(self.weights['W2'], A1) + self.weights['b2']
        A2 = self.relu(Z2)
        Z3 = np.dot(self.weights['W3'], A2) + self.weights['b3']
        A3 = self.tanh(Z3)

        # Return the scaled evaluation score
        return A3[0, 0] * 20

    def evaluate_board(self, board):
        """
        Main method to evaluate the board
        """
        self.board = np.array(board).copy()
        return self.evaluation()

    def get_parameters(self):
        """
        Getter to obtain all parameters
        """
        return self.weights.copy()

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            # Převedení NpzFile na slovník
            self.parameters = {key: parameters[key] for key in parameters.files}
        else:
            # Pokud už je to slovník, uděláme kopii
            self.parameters = parameters.copy()

    def load_params(self,file=""):
        return self.set_parameters(np.load(file))

    def save_params(self, file="params_nn.npy"):
        """
        Save parameters to a file
        """
        parameters = self.get_parameters()
        np.save(file, parameters)


# Example usage
if __name__ == "__main__":
    engine = LewaEngine()
    board_example = [[1, 0, -1, 0, 1], [0, 1, 0, -1, 0], [-1, 0, 1, 0, -1], [0, -1, 0, 1, 0], [1, 0, -1, 0, 1]]
    score = engine.evaluate_board(board_example)
    print(f"Board evaluation score: {score}")
    engine.save_params("saved_params_nn.npy")
    engine.load_params("saved_params_nn.npy")
