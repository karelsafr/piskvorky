import numpy as np

class tym_dvanact:
    def __init__(self):
        self.v_engine = "0.2.0"
        self.board = None
        self.initialize_parameters()

    def initialize_parameters(self):
        input_size = 25
        hidden_size = 10  # Zvětšili jsme skrytou vrstvu pro lepší výkon

        # Inicializace vah pomocí He inicializace pro ReLU
        self.W = np.random.randn(hidden_size, input_size) * np.sqrt(2. / input_size)
        self.b = np.zeros((hidden_size, 1))  # Inicializace biasů na nulu

        self.parameters = {
            'w': self.W,
            'b': self.b
        }

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        for param_name in self.parameters:
            mutation_mask = np.random.random(self.parameters[param_name].shape) < mutation_rate
            mutations = np.random.randn(*self.parameters[param_name].shape) * mutation_scale
            self.parameters[param_name] += mutations * mutation_mask

        self.W = self.parameters['w']
        self.b = self.parameters['b']

    def leaky_relu(self, Z, alpha=0.01):
        """Leaky ReLU aktivační funkce pro lepší gradient"""
        return np.maximum(alpha * Z, Z)

    def tanh(self, Z):
        """Tanh aktivační funkce pro výstup v rozsahu (-1, 1)"""
        return np.tanh(Z)

    def evaluation(self):
        X = self.board.reshape(-1, 1)

        # Dopředný průchod neuronovou sítí
        Z1 = np.dot(self.W, X) + self.b
        A1 = self.leaky_relu(Z1)
        Z2 = np.sum(A1, axis=0, keepdims=True)
        A2 = self.tanh(Z2)  # Použití tanh místo sigmoid

        # Škálování výstupu pro vyhnutí se remízám
        return A2[0, 0] * 25  # Škálování na rozsah (-25, 25)

    def evaluate_board(self, board):
        self.board = board.copy()
        return self.evaluation()

    def get_parameters(self):
        return self.parameters.copy()

    def set_parameters(self, parameters):
        if isinstance(parameters, np.lib.npyio.NpzFile):
            self.parameters = {key.lower(): parameters[key] for key in parameters.files}
        else:
            self.parameters = {key.lower(): value for key, value in parameters.items()}

        self.W = self.parameters['w']
        self.b = self.parameters['b']

    def load_params(self, file=""):
        return self.set_parameters(np.load(file))

