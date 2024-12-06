import numpy as np

class tym_jedna:
    def __init__(self):
        self.v_engine = "0.3.0"
        self.board = None
        self.initialize_parameters()

    def initialize_parameters(self):
        # Parameters are no longer needed for a learned model, but we'll keep the structure.
        self.parameters = {
            'W': np.zeros((1,25)),  # Dummy placeholders
            'b': np.zeros((1,1))
        }

    def mutate(self, mutation_rate=0.1, mutation_scale=0.1):
        # No-op, as we don't need mutation for a threat-based solver.
        pass

    def sigmoid(self, Z):
        # Not needed for threat-based evaluation, but we'll keep it for API consistency
        return 1 / (1 + np.exp(-Z))

    def evaluation(self):
        """
        Perform a threat-based evaluation on self.board.
        Returns a score: large positive if 'X' (assume my symbol = 1) is winning or can force a win,
        large negative if 'O' (opp symbol = -1) is winning or can force a win.
        Otherwise returns a heuristic score.
        """
        board = self.board
        my_symbol = 1
        opp_symbol = -1

        # 1. Check for immediate wins
        val = self._check_terminal(board, my_symbol, opp_symbol)
        if val is not None:
            return val

        # 2. Check for forced wins / double threats
        # If my_symbol can create a double-threat (two moves that each lead to a win),
        # we give a huge positive score. If opponent can, we give a huge negative score.

        forced_val = self._check_forced_situations(board, my_symbol, opp_symbol)
        if forced_val is not None:
            return forced_val

        # 3. Fallback heuristic if no forced situation detected.
        return self._heuristic_evaluation(board, my_symbol, opp_symbol)

    def evaluate_board(self, board):
        """Main method for external calls."""
        self.board = board.copy()
        return self.evaluation()

    def get_parameters(self):
        """Getter for parameters."""
        return self.parameters.copy()

    def set_parameters(self, parameters):
        """Setter for parameters (not actually used here)."""
        if isinstance(parameters, np.lib.npyio.NpzFile):
            self.parameters = {key: parameters[key] for key in parameters.files}
        else:
            self.parameters = parameters.copy()

    def load_params(self, file=""):
        return self.set_parameters(np.load(file))

    # ------------------- Internal Helper Methods -------------------

    def _check_terminal(self, board, my_symbol, opp_symbol):
        # Check if someone already won
        # If my_symbol has 5 in a row: return a large positive
        # If opp_symbol has 5 in a row: return a large negative

        if self._has_five_in_a_row(board, my_symbol):
            return 999999
        if self._has_five_in_a_row(board, opp_symbol):
            return -999999
        return None

    def _check_forced_situations(self, board, my_symbol, opp_symbol):
        # Check if we can create a double threat next move.
        # For simplicity, we’ll check if:
        # - my_symbol has a move that immediately leads to a 5-in-a-row (already handled above),
        # - or can create a position where next turn there will be two separate winning moves.
        #
        # This is a simplification: we’ll look for moves that create multiple lines of length 4 that are open.

        my_threats = self._count_near_wins(board, my_symbol)
        opp_threats = self._count_near_wins(board, opp_symbol)

        # If my_threats > 1, I have a double threat formation
        if my_threats > 1:
            return 500000  # large positive
        if opp_threats > 1:
            return -500000 # large negative

        return None

    def _heuristic_evaluation(self, board, my_symbol, opp_symbol):
        # A fallback heuristic similar to what we discussed:
        # Count lines of length 2,3,4 for each player, reward/punish accordingly.

        lines = self._generate_all_lines(board)

        my_score = 0
        opp_score = 0

        # Tune scoring parameters if desired
        four_score = 1000
        three_score = 100
        two_score = 10

        for line in lines:
            if opp_symbol not in line:
                count_my = np.count_nonzero(line == my_symbol)
                if count_my == 4:
                    my_score += four_score
                elif count_my == 3:
                    my_score += three_score
                elif count_my == 2:
                    my_score += two_score
            if my_symbol not in line:
                count_opp = np.count_nonzero(line == opp_symbol)
                if count_opp == 4:
                    opp_score += four_score
                elif count_opp == 3:
                    opp_score += three_score
                elif count_opp == 2:
                    opp_score += two_score

        return my_score - opp_score

    def _has_five_in_a_row(self, board, symbol):
        lines = self._generate_all_lines(board)
        for line in lines:
            if np.sum(line == symbol) == 5:
                return True
        return False

    def _count_near_wins(self, board, symbol):
        # Count how many lines of length 4 that can become 5 next move
        # Lines that are 4 of 'symbol' and 1 empty slot (0), no opponent symbol present
        count = 0
        lines = self._generate_all_lines(board)
        for line in lines:
            if not (symbol * -1 in line):  # no opponent symbol in that line
                if np.count_nonzero(line == symbol) == 4 and np.count_nonzero(line == 0) == 1:
                    count += 1
        return count

    def _generate_all_lines(self, board):
        # All 5-length lines in a 5x5: rows, columns, main diag, anti diag
        lines = []
        # rows
        for r in range(5):
            lines.append(board[r,:])
        # cols
        for c in range(5):
            lines.append(board[:,c])
        # main diag
        lines.append(np.array([board[i,i] for i in range(5)]))
        # anti diag
        lines.append(np.array([board[i,4-i] for i in range(5)]))
        return lines
