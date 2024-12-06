import numpy as np
from scipy.stats import entropy

class tym_harmonie:
    def __init__(self):
        self.v_engine = "0.0.3"
        self.board = None

        # Inicializace parametrů lineární regrese
        self.initialize_parameters()

    def initialize_parameters(self):
        # Pro 5x5 desku máme 25 vstupů
        input_size = 25

        input_size = 25+10  # For a 5x5 board
        #hidden_layer_size = 10 # default
        hidden_layer_size_1 = 32
        hidden_layer_size_2 = 16
        output_size = 1  # Single output for evaluation
        
        # Parameters for input to hidden layer
        self.W1 = np.random.randn(hidden_layer_size_1, input_size) * 0.01
        self.b1 = np.zeros((hidden_layer_size_1, 1))

        # Parameters for hidden1 to hidden2 layer
        self.W2 = np.random.randn(hidden_layer_size_2, hidden_layer_size_1) * 0.01
        self.b2 = np.zeros((hidden_layer_size_2, 1))

        # Parameters for hidden2 to output layer
        self.W3 = np.random.randn(output_size, hidden_layer_size_2) * 0.01
        self.b3 = np.zeros((output_size, 1))


        # Uložení parametrů do slovníku pro konzistenci s původním API
        self.parameters = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3
        }

    def mutate(self, mutation_rate=0.15, mutation_scale=0.15): # lze upravit parametry ale nastavit defaultní
        """
        Mutace parametrů lineární regrese
        mutation_rate: pravděpodobnost mutace každého parametru
        mutation_scale: síla mutace
        """
        for param_name in self.parameters:
            mutation_mask = np.random.random(self.parameters[param_name].shape) < mutation_rate
            mutations = np.random.randn(*self.parameters[param_name].shape) * mutation_scale
            self.parameters[param_name] += mutations * mutation_mask


        # Update references
        self.W1 = self.parameters['W1']
        self.b1 = self.parameters['b1']
        self.W2 = self.parameters['W2']
        self.b2 = self.parameters['b2']
        self.W3 = self.parameters['W3']
        self.b3 = self.parameters['b3']


        self.parameters = {
            'W1': self.W1, 'b1': self.b1,
            'W2': self.W2, 'b2': self.b2,
            'W3': self.W3, 'b3': self.b3
        }

    def sigmoid(self, Z):
        """Sigmoid aktivační funkce pro normalizaci výstupu"""
        return 1 / (1 + np.exp(-Z))

    def relu(self, Z):
        """ReLU activation function."""
        return np.maximum(0, Z)

    def evaluation(self): # nutná metoda
        """
        Evaluace desky pomocí lineární regrese
        Forward propagation through the network.
        """

        # Horizontal sums (row-wise)
        horizontal_sums = self.board.sum(axis=1)

        # Vertical sums (column-wise)
        vertical_sums = self.board.sum(axis=0)

        # Add sums vectors to flattened board matrix
        X = np.concatenate((self.board.ravel(), horizontal_sums, vertical_sums))
        X = X.reshape(-1, 1)
        
        # Compute hidden layer activations
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)

        # Compute output layer activations
        Z2 = np.dot(self.W2, A1) + self.b2
        A2 = self.relu(Z2)

        # Compute output layer activations
        Z3 = np.dot(self.W3, A2) + self.b3
        A3 = self.sigmoid(Z3)

        # print(f"X shape: {X.shape}")
        # print(f"Z1 shape: {Z1.shape}, A1 shape: {A1.shape}")
        # print(f"W1 shape: {self.W1.shape}, b1 shape: {self.b1.shape}")
        # print(f"Z2 shape: {Z2.shape}, A2 shape: {A2.shape}")
        # print(f"W2 shape: {self.W2.shape}, b2 shape: {self.b2.shape}")
        # print(f"Z3 shape: {Z3.shape}, A3 shape: {A3.shape}")
        # print(f"W3 shape: {self.W3.shape}, b3 shape: {self.b3.shape}")
        
        value = A3.item()

        return value

    def evaluate_board(self, board):
        """Hlavní metoda pro evaluaci desky"""
        self.board = board.copy()
        return self.evaluation()

    def get_parameters(self):
        """Getter pro získání všech parametrů"""
        return self.parameters.copy()

    def set_parameters(self, parameters):
        """Setter pro nastavení všech parametrů"""
        if isinstance(parameters, np.lib.npyio.NpzFile):
            # Převedení NpzFile na slovník
            self.parameters = {key: parameters[key] for key in parameters.files}
        else:
            # Pokud už je to slovník, uděláme kopii
            self.parameters = parameters.copy()

        self.W1 = self.parameters['W1']
        self.b1 = self.parameters['b1']
        self.W2 = self.parameters['W2']
        self.b2 = self.parameters['b2']
        self.W3 = self.parameters['W3']
        self.b3 = self.parameters['b3']

    def load_params(self,file=""):
        return self.set_parameters(np.load(file))
