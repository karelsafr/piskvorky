import numpy as np
from scipy.signal import convolve2d


class EngineCNNPrdCF:
    def __init__(self):
        self.v_engine = "0.0.2"
        self.board = None
        self.initialize_parameters()

    def initialize_parameters(self):
        self.conv_filters = np.random.randn(8, 3, 3) * 0.01
        self.W_fc = np.random.randn(4, 8) * 0.01
        self.b_fc = np.zeros((4, 1))
        self.parameters = {
            'conv_filters': self.conv_filters,
            'W_fc': self.W_fc,
            'b_fc': self.b_fc
        }

    def crossover(self, other_engine):
        """Combine parameters from two engines."""
        child = EngineCNNPrdCF()
        for param_name in self.parameters:
            param_self = self.parameters[param_name]
            param_other = other_engine.parameters[param_name]
            crossover_mask = np.random.rand(*param_self.shape) < 0.5
            child.parameters[param_name] = np.where(
                crossover_mask, param_self, param_other
            )
        child.update_parameters()
        return child

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        for param_name in self.parameters:
            param = self.parameters[param_name]
            mutation_mask = np.random.rand(*param.shape) < mutation_rate
            mutations = np.random.randn(*param.shape) * mutation_scale
            self.parameters[param_name] += mutations * mutation_mask
        self.update_parameters()

    def update_parameters(self):
        self.conv_filters = self.parameters['conv_filters']
        self.W_fc = self.parameters['W_fc']
        self.b_fc = self.parameters['b_fc']

    def relu(self, Z):
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def evaluation(self):
        X = self.board
        conv_outputs = [convolve2d(X, f, mode='valid') for f in self.conv_filters]
        conv_outputs = np.array(conv_outputs).reshape(-1, 1)
        Z = np.dot(self.W_fc, conv_outputs) + self.b_fc
        A = self.sigmoid(Z)
        return (A[0, 0] - 0.5) * 20

    def evaluate_board(self, board):
        self.board = board
        return self.evaluation()

    def get_parameters(self):
        return self.parameters.copy()

    def set_parameters(self, parameters):
        self.parameters = parameters.copy()
        self.update_parameters()

    def load_params(self, file=""):
        try:
            data = np.load(file)
            # Print available keys for debugging
            print(f"Available keys in file: {data.files}")

            # More flexible loading that handles both old and new parameter formats
            if 'conv_filters' in data.files:
                self.parameters = {
                    'conv_filters': data['conv_filters'],
                    'W_fc': data['W_fc'],
                    'b_fc': data['b_fc']
                }
            else:
                # Handle legacy format or different parameter structure
                self.parameters = {key: data[key] for key in data.files}

            self.update_parameters()
        except Exception as e:
            print(f"Error loading parameters: {e}")
            # Initialize new parameters if loading fails
            self.initialize_parameters()

    def save_params(self, file=""):
        np.savez(file,
                 conv_filters=self.conv_filters,
                 W_fc=self.W_fc,
                 b_fc=self.b_fc)