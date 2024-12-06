import numpy as np

class tym_imes:
    def __init__(self, input_size=25, hidden_size=64, output_size=1, learning_rate=0.01):
        self.v_engine = "0.2.0"
        self.board = None
        self.learning_rate = learning_rate

        # Default parameters for initialization
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize the parameters of the neural network
        self.initialize_parameters()

    def initialize_parameters(self, input_size=None, hidden_size=None, output_size=None):
        input_size = input_size if input_size is not None else self.input_size
        hidden_size = hidden_size if hidden_size is not None else self.hidden_size
        output_size = output_size if output_size is not None else self.output_size

        self.parameters = {
            'W1': np.random.randn(hidden_size, input_size) * 0.01,
            'b1': np.zeros((hidden_size, 1)),
            'W2': np.random.randn(output_size, hidden_size) * 0.01,
            'b2': np.zeros((output_size, 1))
        }

    def mutate(self, mutation_rate=0.05, mutation_scale=0.1):
        """Mutate the parameters of the model using numpy random methods"""
        for param_name, param_value in self.parameters.items():
            # Randomly select elements for mutation using numpy
            mutation_mask = np.random.rand(*param_value.shape) < mutation_rate
            # Create random changes (Gaussian noise)
            mutations = np.random.randn(*param_value.shape) * mutation_scale
            # Apply mutations
            self.parameters[param_name] += mutations * mutation_mask

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def forward_propagation(self, X):
        W1, b1, W2, b2 = self.parameters.values()

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = self.sigmoid(Z2)

        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}
        return A2, cache

    def compute_loss(self, A2, Y):
        """Compute the cross-entropy loss using numpy"""
        m = Y.shape[1]
        loss = -np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2)) / m
        return loss

    def evaluation(self):
        """Evaluates the board using the neural network"""
        X = self.board.reshape(-1, 1)
        A2, _ = self.forward_propagation(X)
        return (A2[0, 0] - 0.5) * 20  # Scale the result

    def evaluate_board(self, board):
        """Main evaluation method for the board"""
        self.board = board.copy()
        return self.evaluation()

    def get_parameters(self):
        return self.parameters.copy()

    def set_parameters(self, parameters):
        self.parameters = parameters.copy()

    def save_params(self, filepath):
        np.savez(filepath, **self.parameters)

    def load_params(self, filepath):
        loaded = np.load(filepath)
        self.parameters = {key: loaded[key] for key in loaded.files}

    def cost(self, board, good_move):
        """Calculate the cost (error) of the network's prediction"""
        outputs, _ = self.forward_propagation(board)
        cost = 0
        for i in range(self.output_size):
            good_output = 1 if i == good_move else 0
            cost += (good_output - outputs[i]) ** 2
        return cost

    def accuracy(self, situations):
        """Calculate the accuracy of the network on a set of situations"""
        correct = 0
        total = len(situations)
        for situation in situations:
            board, good_move = situation
            if self.find_move(board) == good_move:
                correct += 1
        return correct / total

    def find_move(self, board):
        """Find the network's preferred move"""
        outputs, _ = self.forward_propagation(board)
        max_index = np.argmax(outputs)
        return max_index

    def train_gradient(self, situations, learn_rate=0.01):
        """Train the network using gradient descent"""
        gradient = self.calculate_gradient(situations)
        for param_name, param_value in self.parameters.items():
            for i in range(param_value.shape[0]):
                for j in range(param_value.shape[1]):
                    self.parameters[param_name][i, j] -= learn_rate * gradient[param_name][i, j]

    def calculate_gradient(self, situations):
        """Approximate the partial derivatives of cost wrt a node's weight"""
        gradient = {key: np.zeros_like(value) for key, value in self.parameters.items()}
        for situation in situations:
            board, good_move = situation
            current_loss = self.cost(board, good_move)
            for param_name, param_value in self.parameters.items():
                for i in range(param_value.shape[0]):
                    for j in range(param_value.shape[1]):
                        param_value[i, j] += 0.001
                        new_loss = self.cost(board, good_move)
                        gradient[param_name][i, j] = (new_loss - current_loss) / 0.001
                        param_value[i, j] -= 0.001  # Revert change
        return gradient

    def train(self, situations, epochs=1000):
        """Train the model over several epochs"""
        for epoch in range(epochs):
            self.train_gradient(situations, learn_rate=self.learning_rate)
            if epoch % 100 == 0:
                acc = self.accuracy(situations)
                print(f"Epoch {epoch}/{epochs}, Accuracy: {acc * 100:.2f}%")
            # Optionally mutate parameters every epoch
            self.mutate(mutation_rate=0.01, mutation_scale=0.1)
