import numpy as np

class tym_fazolky:
    def __init__(self):
        self.v_engine = "0.0.3"
        self.board = None
        self.initialize_parameters()

    def initialize_parameters(self):
        # Počet vstupů odpovídá velikosti desky (5x5)
        input_size = 25
        self.W = np.random.randn(1, input_size) * 0.01  # Váhy
        self.b = np.zeros((1, 1))  # Bias
        self.parameters = {'W': self.W, 'b': self.b}

    def sigmoid(self, Z):
        """Sigmoid aktivační funkce pro normalizaci výstupu."""
        return 1 / (1 + np.exp(-Z))

    def evaluation(self):
        """
        Evaluace desky pomocí lineární regrese.
        Vrací skóre optimalizované pro útok i obranu.
        """
        X = self.board.reshape(-1, 1)  # Převedení desky do vektoru
        Z = np.dot(self.W, X) + self.b  # Lineární regrese
        A = self.sigmoid(Z)  # Normalizace výstupu
        return (A[0, 0] - 0.5) * 20

    def evaluate_board(self, board):
        """
        Vyhodnocení skóre pro daný stav desky.
        """
        self.board = board.copy()
        return self.evaluation()

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        """
        Mutace parametrů lineární regrese pro průzkum prostoru řešení.
        """
        for param_name in self.parameters:
            mutation_mask = np.random.random(self.parameters[param_name].shape) < mutation_rate
            mutations = np.random.randn(*self.parameters[param_name].shape) * mutation_scale
            self.parameters[param_name] += mutations * mutation_mask
        self.W = self.parameters['W']
        self.b = self.parameters['b']

    def train(self, training_data, epochs=1000, learning_rate=0.01):
        """
        Trénování modelu na trénovacích datech.
        :param training_data: seznam dvojic (board, score)
        :param epochs: počet iterací tréninku
        :param learning_rate: krok učení
        """
        for epoch in range(epochs):
            total_loss = 0
            for board, target_score in training_data:
                X = board.reshape(-1, 1)  # Zploštění desky
                Z = np.dot(self.W, X) + self.b
                A = self.sigmoid(Z)
                error = A[0, 0] - target_score  # Chyba
                total_loss += error ** 2

                # Gradient sestupu
                dZ = error * A * (1 - A)
                dW = np.dot(dZ, X.T)
                db = dZ

                # Aktualizace parametrů
                self.W -= learning_rate * dW
                self.b -= learning_rate * db

            # Výpis ztráty pro každou epochu
            if (epoch + 1) % (epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    def defensive_score(self, board, opponent_marker=1):
        """
        Vypočítá skóre ohrožení pro obranné tahy.
        Identifikuje místa, kde by soupeř mohl vyhrát.
        """
        danger_zones = 0
        size = int(np.sqrt(board.size))  # Velikost desky
        board = board.reshape((size, size))
        for row in range(size):
            if np.sum(board[row, :] == opponent_marker) == size - 1 and np.sum(board[row, :] == 0) == 1:
                danger_zones += 1
        for col in range(size):
            if np.sum(board[:, col] == opponent_marker) == size - 1 and np.sum(board[:, col] == 0) == 1:
                danger_zones += 1
        if np.sum(np.diag(board) == opponent_marker) == size - 1 and np.sum(np.diag(board) == 0) == 1:
            danger_zones += 1
        if np.sum(np.diag(np.fliplr(board)) == opponent_marker) == size - 1 and np.sum(np.diag(np.fliplr(board)) == 0) == 1:
            danger_zones += 1
        return danger_zones

    def evaluate_move(self, board, move, player):
        """
        Vyhodnotí tah na základě útoku i obrany.
        """
        board[move] = player
        offensive_score = self.evaluate_board(board)
        defensive_score = -self.defensive_score(board, opponent_marker=-player)
        board[move] = 0  # Vrácení do původního stavu
        return offensive_score + defensive_score

    def find_best_move(self, board, player):
        """
        Najde nejlepší tah kombinací útoku a obrany.
        """
        best_score = -np.inf
        best_move = None
        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row, col] == 0:  # Pokud je pole volné
                    score = self.evaluate_move(board, (row, col), player)
                    if score > best_score:
                        best_score = score
                        best_move = (row, col)
        return best_move

    def get_parameters(self):
        """Vrátí parametry modelu."""
        return self.parameters.copy()

    def set_parameters(self, parameters):
        """Nastaví parametry modelu."""
        self.parameters = parameters
        self.W = self.parameters['W']
        self.b = self.parameters['b']

    def load_params(self, file=""):
        """Načte parametry ze souboru."""
        self.set_parameters(np.load(file))