import numpy as np
import pygame
from GameBoard import GameBoard
from Player import Player
from colors import *
import time


class HumanVsAI:
    def __init__(self, engine_class, engine_datafile=None, display_game=True, ai_starts=False):
        """
        Initialize the human vs AI game
        engine_class: Engine class to use (e.g., EngineLinear, EngineConvo)
        engine_datafile: Optional file path to load trained engine parameters
        display_game: Whether to show visualization (should be True for human play)
        ai_starts: Whether AI makes the first move
        """
        self.display_game = display_game
        self.ai_starts = ai_starts

        # Initialize players
        if ai_starts:
            self.ai_player = Player(+1, engine_class, engine_datafile)
            self.human_mark = -1
        else:
            self.ai_player = Player(-1, engine_class, engine_datafile)
            self.human_mark = +1

        # Initialize pygame
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Human vs AI")
        self.clock = pygame.time.Clock()
        self.board_display = GameBoard(50, 50, self.screen)
        self.font = pygame.font.Font(None, 36)

        # Game state
        self.board = np.zeros((5, 5), dtype=int)
        self.game_over = False
        self.winner = None

    def handle_mouse_click(self, pos):
        """Convert mouse position to board coordinates and make move if valid"""
        # Convert screen coordinates to board coordinates
        board_x = (pos[0] - self.board_display.x) // self.board_display.size
        board_y = (pos[1] - self.board_display.y) // self.board_display.size

        # Check if click is within board bounds
        if (0 <= board_x < 5 and 0 <= board_y < 5 and
                self.board[board_x, board_y] == 0):
            self.board[board_x, board_y] = self.human_mark
            return True
        return False

    def check_winner(self):
        """Check if there's a winner or draw"""
        if self.ai_player.check_win(1):
            return 1
        elif self.ai_player.check_win(-1):
            return -1
        elif not any(0 in row for row in self.board):
            return 0
        return None

    def update_display(self):
        """Update the game display"""
        self.screen.fill(WHITE)

        # Update board display
        self.board_display.set_game(type('GameData', (), {
            'board': self.board,
            'player1name': "Human" if self.human_mark == 1 else "AI",
            'player2name': "AI" if self.human_mark == 1 else "Human",
            'player1score': 0,
            'player2score': 0,
            'active': not self.game_over
        }))
        self.board_display.draw()

        # Display game status
        status_text = "Your turn" if not self.game_over else self.get_game_over_text()
        text_surface = self.font.render(status_text, True, BLACK)
        self.screen.blit(text_surface, (400, 50))

        pygame.display.flip()

    def get_game_over_text(self):
        """Get the game over message"""
        if self.winner == self.human_mark:
            return "You won!"
        elif self.winner == -self.human_mark:
            return "AI won!"
        else:
            return "Draw!"

    def run_game(self):
        """Main game loop"""
        current_player = 1  # 1 for first player, -1 for second player

        # If AI starts, make its first move
        if self.ai_starts:
            self.ai_player.play(self.board)
            self.board = self.ai_player.get_board()
            current_player = -1

        while not self.game_over:
            self.update_display()

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

                # Handle mouse clicks during human turn
                if event.type == pygame.MOUSEBUTTONDOWN and current_player == self.human_mark:
                    if self.handle_mouse_click(event.pos):
                        self.winner = self.check_winner()
                        if self.winner is not None:
                            self.game_over = True
                            break

                        # AI's turn
                        self.ai_player.play(self.board)
                        self.board = self.ai_player.get_board()

                        self.winner = self.check_winner()
                        if self.winner is not None:
                            self.game_over = True

            self.clock.tick(30)

        # Show final state
        self.update_display()

        # Wait for window close
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False

        pygame.quit()
        return self.winner


if __name__ == "__main__":
    from Team_ABO_Engine import TeamABOEngine

    game = HumanVsAI(
        engine_class=TeamABOEngine,
        engine_datafile="./temp_engines/winner_engine.npz",  # Optional: load trained parameters
        display_game=True,
        ai_starts=False  # AI makes first move
    )

    result = game.run_game()

