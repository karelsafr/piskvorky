import numpy as np
import itertools
from main_komparace import EngineComparison
import pandas as pd
from pathlib import Path

class TournamentManager:
    def __init__(self, engines_config, games_per_match=5, display_game=True, delay=0.1):
        self.engines_config = engines_config
        self.games_per_match = games_per_match
        self.display_game = display_game
        self.delay = delay

        # Vytvoření popisnějších názvů pro enginy
        self.engine_names = [
            f"{engine_class.__name__}_{i + 1}"
            for i, (engine_class, _) in enumerate(engines_config)
        ]

        # Inicializace výsledkové matice s popisnějšími názvy
        self.results = pd.DataFrame(0,
                                    index=self.engine_names,
                                    columns=self.engine_names)

        self.detailed_results = {}

    def run_tournament(self):
        """Spustí kompletní turnaj - každý engine hraje proti všem ostatním"""
        # Získání všech možných párů enginů
        pairs = list(itertools.combinations(range(len(self.engines_config)), 2))
        total_matches = len(pairs)

        print(f"Starting tournament with {len(self.engines_config)} engines")
        print(f"Total matches to play: {total_matches}")

        for idx, (i, j) in enumerate(pairs):
            print(f"\nMatch {idx + 1}/{total_matches}:")
            engine1_class, engine1_file = self.engines_config[i]
            engine2_class, engine2_file = self.engines_config[j]

            print(f"{self.engine_names[i]} vs {self.engine_names[j]}")

            # Spuštění porovnání mezi dvěma enginy
            comparison = EngineComparison(
                engine1_class=engine1_class,
                engine1_datafile=engine1_file,
                engine2_class=engine2_class,
                engine2_datafile=engine2_file,
                num_games=self.games_per_match,
                display_game=self.display_game,
                delay=self.delay
            )
            comparison.run_comparison()

            # Uložení výsledků
            self.results.iloc[i, j] = comparison.engine1_wins
            self.results.iloc[j, i] = comparison.engine2_wins

            # Uložení detailních výsledků
            match_key = f"{self.engine_names[i]}_vs_{self.engine_names[j]}"
            self.detailed_results[match_key] = {
                'engine1_wins': comparison.engine1_wins,
                'engine2_wins': comparison.engine2_wins,
                'draws': comparison.draws,
                'win_rate1': comparison.engine1_wins / self.games_per_match * 100,
                'win_rate2': comparison.engine2_wins / self.games_per_match * 100,
                'draw_rate': comparison.draws / self.games_per_match * 100
            }

        self.display_tournament_results()
        self.save_results()

    def display_tournament_results(self):
        print("\n=== Tournament Results ===")

        # Zobrazení výsledků podle typu enginu
        engine_types = set(name.split('_')[0] for name in self.engine_names)

        print("\nResults by Engine Type:")
        for engine_type in engine_types:
            type_engines = [name for name in self.engine_names if name.startswith(engine_type)]
            print(f"\n{engine_type} Engines:")
            for engine in type_engines:
                total_wins = self.results.loc[engine].sum()
                total_games = total_wins + self.results[engine].sum()
                win_rate = (total_wins / total_games * 100) if total_games > 0 else 0
                print(f"  {engine}: {total_wins} wins ({win_rate:.1f}% win rate)")

        print("\nDetailed Match Results:")
        for match, results in self.detailed_results.items():
            print(f"\n{match}:")
            print(f"  Wins: {results['engine1_wins']} vs {results['engine2_wins']}")
            print(f"  Draws: {results['draws']}")
            print(f"  Win Rates: {results['win_rate1']:.1f}% vs {results['win_rate2']:.1f}%")

    def save_results(self):
        """Uložení výsledků do souborů"""
        self.results.to_csv('tournament_results_matrix.csv')
        detailed_df = pd.DataFrame.from_dict(self.detailed_results, orient='index')
        detailed_df.to_csv('tournament_results_detailed.csv')

        print("\nResults have been saved to:")
        print("- tournament_results_matrix.csv")
        print("- tournament_results_detailed.csv")


if __name__ == "__main__":
    from Team_ABO_Engine import TeamABOEngine
    from EngineLinear import EngineLinear

    # Konfigurace s enginy
    engine_configs = [
        # Lineární enginy
        (TeamABOEngine, "./temp_engines/winner_engine.npz"),
        (EngineLinear, "./temp_engines/LinEngine1.npz"),]

    # Konfigurace turnaje
    tournament = TournamentManager(
        engines_config=engine_configs,
        games_per_match=5,
        display_game=True,
        delay=0.1 # seconds to move
    )

    # Spuštění turnaje
    tournament.run_tournament()