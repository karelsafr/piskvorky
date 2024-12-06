import numpy as np
from scipy.stats import entropy

class TeamABOEngine:
    def __init__(self):
        self.v_engine = "0.0.1"
        self.board = None
        self.initialize_parameters()

    def initialize_parameters(self):
        # Pro 5x5 desku máme 25 vstupů
        input_size = 25

        # Váhy pro lineární regresi
        self.W = np.random.randn(1, input_size) * 0.01
        self.b = np.zeros((1, 1))

        # Uložení parametrů do slovníku pro konzistenci s původním API
        self.parameters = {
            'W': self.W,
            'b': self.b
        }

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        """
        Mutace parametrů lineární regrese
        mutation_rate: pravděpodobnost mutace každého parametru
        mutation_scale: síla mutace
        """
        for param_name in self.parameters:
            mutation_mask = np.random.random(self.parameters[param_name].shape) < mutation_rate
            mutations = np.random.randn(*self.parameters[param_name].shape) * mutation_scale
            self.parameters[param_name] += mutations * mutation_mask

        # Aktualizace referencí na parametry
        self.W = self.parameters['W']
        self.b = self.parameters['b']

    def sigmoid(self, Z):
        """Sigmoid aktivační funkce pro normalizaci výstupu"""
        return 1 / (1 + np.exp(-Z))

    def evaluation(self):
        """
        Evaluace desky pomocí lineární regrese
        """
        # Zploštění desky do vektoru
        X = self.board.reshape(-1, 1)

        # Lineární regrese
        Z = np.dot(self.W, X) + self.b

        # Sigmoid pro normalizaci výstupu
        A = self.sigmoid(Z)
        return (A[0, 0] - 0.5) * 20

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

        self.W = self.parameters['W']
        self.b = self.parameters['b']

    def load_params(self,file=""):
        return self.set_parameters(np.load(file))
