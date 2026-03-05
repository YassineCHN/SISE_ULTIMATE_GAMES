# main.py — point d'entrée pour lancer un jeu depuis le dashboard
import sys
import pygame
from games.reflex_game import ReflexGame
from games.labyrinth_game import LabyrinthGame
from games.shooter_game import TwinStickShooter
from games.racing_game import RacingGame

GAMES = {
    "reflex":    ReflexGame,
    "labyrinth": LabyrinthGame,
    "shooter":   TwinStickShooter,
    "racing":    RacingGame,
}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python main.py <game_id> <player_name>")
        sys.exit(1)

    game_id     = sys.argv[1]
    player_name = sys.argv[2]

    if game_id not in GAMES:
        print(f"Jeu inconnu : {game_id}. Choix : {list(GAMES.keys())}")
        sys.exit(1)

    print(f"🎮 Lancement de '{game_id}' pour '{player_name}'")
    game = GAMES[game_id](player_name=player_name)
    features = game.run()
    print(f"✅ Session terminée — score : {features.score}")