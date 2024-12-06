import numpy as np


class tym_team:
    def init(self):
        self.v_engine = "0.0.1"  # Version tracking
        self.board = None  # This will be the 5x5 game board
        self.initialize_parameters()

    def initialize_parameters(self):
        """
        Initialize CNN parameters:
        Filters: 2 filters of size 3x3
        Weights: Fully connected layer that flattens the convolution outputs
        """
        self.filters = np.random.randn(2, 3, 3) * 0.1  # 2 filters of size 3x3
        self.weights = np.random.randn(2 * (5 - 2) ** 2, 1) * 0.1  # Fully connected layer weights
        self.bias = np.zeros((1, 1))  # Bias for the fully connected layer

        # Store parameters in a dictionary for framework compatibility
        self.parameters = {
            'filters': self.filters,
            'weights': self.weights,
            'bias': self.bias
        }

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        """
        Mutate the CNN parameters with given mutation rate and scale
        """
        for key in self.parameters:
            param = self.parameters[key]
            mutation_mask = np.random.random(param.shape) < mutation_rate
            mutations = np.random.randn(*param.shape) * mutation_scale
            self.parameters[key] += mutations * mutation_mask

        # Update internal references for filters, weights, and bias
        self.filters = self.parameters['filters']
        self.weights = self.parameters['weights']
        self.bias = self.parameters['bias']

    def convolve(self, board, filter):
        """
        Perform 2D convolution operation on the board with the given filter.
        This method applies the filter over the 3x3 patches of the 5x5 board.
        """
        result = []
        # Perform convolution for the 5x5 board (valid padding, 3x3 filter)
        for i in range(board.shape[0] - 2):  # Only iterate within valid range
            for j in range(board.shape[1] - 2):
                sub_board = board[i:i + 3, j:j + 3]  # 3x3 patch
                result.append(np.sum(sub_board * filter))  # Element-wise multiplication and summing
        return np.array(result)

    def evaluation(self):
        """
        Evaluate the current board using CNN.
        It returns a numerical value: positive favoring player 1, negative favoring player -1.
        """
        if self.board is None:
            raise ValueError("Board is not set for evaluation.")

        # Apply each filter to the board
        conv_outputs = [self.convolve(self.board, f) for f in self.filters]

        # Flatten the convolution outputs to pass through a fully connected layer
        flattened = np.concatenate(conv_outputs)

        # Apply the fully connected layer (weights and bias)
        score = np.dot(flattened, self.weights) + self.bias
        return score.item()  # Convert from array to scalar value

    def evaluate_board(self, board):
        """Set the board and evaluate it."""
        self.board = board.copy()  # Ensure the board is not modified outside this method
        return self.evaluation()

    def get_parameters(self):
        """Return a copy of the parameters."""
        return {key: param.copy() for key, param in self.parameters.items()}

    def set_parameters(self, parameters):
        """Set the parameters from a dictionary."""
        self.parameters = {key: param.copy() for key, param in parameters.items()}

        # Update internal references for filters, weights, and bias
        self.filters = self.parameters['filters']
        self.weights = self.parameters['weights']
        self.bias = self.parameters['bias']

    def load_params(self, file=""):
        """Load parameters from a file."""
        loaded_params = np.load(file, allow_pickle=True)
        self.set_parameters({key: loaded_params[key] for key in loaded_params.files})