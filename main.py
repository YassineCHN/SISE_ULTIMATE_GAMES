# main.py — point d'entrée pour lancer un jeu depuis le dashboard
import sys
import argparse
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
    parser = argparse.ArgumentParser(description="Lancer un jeu SISE")
    parser.add_argument("game_id",     nargs="?", help="ID du jeu (positional)")
    parser.add_argument("player_name", nargs="?", help="Nom du joueur (positional)")
    parser.add_argument("--game",   dest="game_flag",   help="ID du jeu")
    parser.add_argument("--player", dest="player_flag", help="Nom du joueur")
    args = parser.parse_args()

    game_id     = args.game_id     or args.game_flag
    player_name = args.player_name or args.player_flag

    if not game_id or not player_name:
        print("Usage: python main.py <game_id> <player_name>")
        print("       python main.py --game <game_id> --player <player_name>")
        sys.exit(1)

    if game_id not in GAMES:
        print(f"Jeu inconnu : {game_id}. Choix : {list(GAMES.keys())}")
        sys.exit(1)

    print(f"🎮 Lancement de '{game_id}' pour '{player_name}'")
    game = GAMES[game_id](player_name=player_name)
    features = game.run()
    print(f"✅ Session terminée — score : {features.score}")