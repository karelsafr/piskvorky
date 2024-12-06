import numpy as np
import pygame
from GameBoard import GameBoard
from Player import Player
from EngineLinear import EngineLinear
from Team_ABO_Engine import TeamABOEngine
import time
from colors import *


class EngineComparison:
    def __init__(self, engine1_class,engine1_datafile, engine2_class,engine2_datafile, num_games=10, display_game=True, delay=0.5):
        """
        Initialize engine comparison
        engine1_class, engine2_class: Engine classes to compare
        num_games: Number of games to play
        display_game: Whether to show visualization
        delay: Delay between moves (seconds) for visualization
        """
        self.engine1_class = engine1_class
        self.engine1_datafile = engine1_datafile
        self.engine2_class = engine2_class
        self.engine2_datafile = engine2_datafile
        self.num_games = num_games
        self.display_game = display_game
        self.delay = delay

        # Statistics
        self.engine1_wins = 0
        self.engine2_wins = 0
        self.draws = 0

        # Initialize pygame if display is enabled
        if self.display_game:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Engine Comparison")
            self.clock = pygame.time.Clock()
            self.board_display = GameBoard(50, 50, self.screen)
            self.font = pygame.font.Font(None, 36)

    def run_comparison(self):
        """Run the comparison between engines"""
        for game_num in range(self.num_games):
            # Initialize players with alternating first moves
            if game_num % 2 == 0:
                player1 = Player(+1, self.engine1_class, self.engine1_datafile)
                player2 = Player(-1, self.engine2_class, self.engine2_datafile)
                first_engine = "Engine 1"
            else:
                player1 = Player(+1, self.engine2_class, self.engine2_datafile)
                player2 = Player(-1, self.engine1_class, self.engine1_datafile)
                first_engine = "Engine 2"

            result = self.play_game(player1, player2, game_num + 1, first_engine)
            self.update_statistics(result, game_num % 2 == 0)

        self.display_final_results()

        if self.display_game:
            # Wait for window close
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        waiting = False
            pygame.quit()

    def play_game(self, player1, player2, game_num, first_engine):
        """Play a single game between two players"""
        move_count = 0

        while True:
            # Player 1's turn
            player1.play(player2.get_board())
            move_count += 1

            if self.display_game:
                self.update_display(player1.get_board(), game_num, first_engine, move_count)
                time.sleep(self.delay)

            if player1.check_win(1):
                return 1
            elif move_count >= 25:  # Board is full
                return 0

            # Player 2's turn
            player2.play(player1.get_board())
            move_count += 1

            if self.display_game:
                self.update_display(player2.get_board(), game_num, first_engine, move_count)
                time.sleep(self.delay)

            if player2.check_win(-1):
                return -1
            elif move_count >= 25:  # Board is full
                return 0

    def update_statistics(self, result, normal_order):
        """Update win/loss statistics"""
        if result == 1:
            if normal_order:
                self.engine1_wins += 1
            else:
                self.engine2_wins += 1
        elif result == -1:
            if normal_order:
                self.engine2_wins += 1
            else:
                self.engine1_wins += 1
        else:
            self.draws += 1

    def update_display(self, board, game_num, first_engine, move_count):
        """Update the game display"""
        self.screen.fill(WHITE)

        # Update board display
        self.board_display.set_game(type('GameData', (), {
            'board': board,
            'player1name': type(self.engine1_class).__name__,
            'player2name': type(self.engine2_class).__name__,
            'player1score': self.engine1_wins,
            'player2score': self.engine2_wins,
            'active': True
        }))
        self.board_display.draw()

        # Display game information
        info_text = [
            f"Game: {game_num}/{self.num_games}",
            f"First Move: {first_engine}",
            f"Move: {move_count}",
            f"Engine 1 ({type(self.engine1_class).__name__}) Wins: {self.engine1_wins}",
            f"Engine 2 ({type(self.engine2_class).__name__}) Wins: {self.engine2_wins}",
            f"Draws: {self.draws}"
        ]

        for i, text in enumerate(info_text):
            text_surface = self.font.render(text, True, BLACK)
            self.screen.blit(text_surface, (400, 50 + i * 40))

        pygame.display.flip()

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

    def display_final_results(self):
        """Display final comparison results"""
        print("\nFinal Results:")
        print(f"Engine 1 ({type(self.engine1_class).__name__}):")
        print(f"  Wins: {self.engine1_wins} ({self.engine1_wins / self.num_games * 100:.1f}%)")
        print(f"Engine 2 ({type(self.engine2_class).__name__}):")
        print(f"  Wins: {self.engine2_wins} ({self.engine2_wins / self.num_games * 100:.1f}%)")
        print(f"Draws: {self.draws} ({self.draws / self.num_games * 100:.1f}%)")


if __name__ == "__main__":
    comparison = EngineComparison(
        engine1_class=EngineLinear,
        engine1_datafile="./temp_engines/LinEngine1.npz",
        engine2_class=TeamABOEngine,
        engine2_datafile="./temp_engines/winner_engine.npz",
        num_games=20,
        display_game=True,
        delay=0.01  # second delay between moves
    )
    comparison.run_comparison()