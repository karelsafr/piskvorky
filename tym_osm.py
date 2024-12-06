import numpy as np
from scipy.stats import entropy

class tym_osm:
    def __init__(self):
        self.v_engine = "0.0.2"
        self.board = None

        # Initialization of CNN parameters
        self.initialize_parameters()

    def initialize_parameters(self):
        # Convolutional layer parameters: Kernel size 3x3, 1 input channel, 8 output channels
        self.W_conv = np.random.randn(3, 3, 1, 8) * 0.01  
        self.b_conv = np.zeros((8,))

        # Fully connected layer parameters adjusted to new flattening size (if unchanged, use 200)
        self.W_fc = np.random.randn(10, 1) * 0.01
        self.b_fc = np.zeros((1, 1))

        # Store parameters in a dictionary
        self.parameters = {
            'W_conv': self.W_conv,
            'b_conv': self.b_conv,
            'W_fc': self.W_fc,
            'b_fc': self.b_fc
        }

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        """Mutate parameters of CNN"""
        for param_name in self.parameters:
            mutation_mask = np.random.random(self.parameters[param_name].shape) < mutation_rate
            mutations = np.random.randn(*self.parameters[param_name].shape) * mutation_scale
            self.parameters[param_name] += mutations * mutation_mask

        # Update references to parameters
        self.update_parameters()

    def sigmoid(self, Z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-Z))

    def evaluation(self):
        """Evaluate board using CNN"""
        # Reshape board to match CNN input
        X = self.board.reshape(5, 5, 1)

        # Apply convolutional layer
        Z_conv = np.zeros((5, 5, 8))
        for i in range(8):  # for each filter
            for x in range(3):  # filter size
                for y in range(3):
                    Z_conv[:, :, i] += np.roll(np.roll(X[:, :, 0], x - 1, axis=0), y - 1, axis=1) * self.W_conv[x, y, 0, i]
            Z_conv[:, :, i] += self.b_conv[i]

        # Activation (ReLU for convolution)
        A_conv = np.maximum(0, Z_conv)

        # Pooling (Max pooling)
        A_pool = A_conv.reshape(5, 5, 2, 4).max(axis=3).max(axis=1)

        # Flatten
        A_flat = A_pool.flatten()

        # Fully connected layer
        Z_fc = np.dot(self.W_fc.T, A_flat) + self.b_fc

        # Sigmoid activation
        A_fc = self.sigmoid(Z_fc)
        return (A_fc[0, 0] - 0.5) * 20

    def evaluate_board(self, board):
        """Main method to evaluate the board"""
        self.board = board.copy()
        return self.evaluation()

    def get_parameters(self):
        """Get all parameters"""
        return self.parameters.copy()

    def set_parameters(self, parameters):
        """Set all parameters"""
        if isinstance(parameters, np.lib.npyio.NpzFile):
            # Convert NpzFile to dictionary
            self.parameters = {key: parameters[key] for key in parameters.files}
        else:
            # If already a dictionary, make a copy
            self.parameters = parameters.copy()

        self.update_parameters()

    def update_parameters(self):
        """Update parameter references after setting or mutating"""
        self.W_conv = self.parameters['W_conv']
        self.b_conv = self.parameters['b_conv']
        self.W_fc = self.parameters['W_fc']
        self.b_fc = self.parameters['b_fc']

    def load_params(self, file=""):
        return self.set_parameters(np.load(file))
