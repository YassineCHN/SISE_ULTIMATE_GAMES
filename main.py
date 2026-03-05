"""
main.py — Point d'entrée principal
Lance un jeu en mode humain ou en mode agent IA.

Usage :
  # Mode humain
  python main.py <game_id> <player_name>

  # Mode agent
  python main.py <game_id> <player_name> --agent <profile_name>
  python main.py <game_id> <player_name> --agent <profile_name> --data <path_csv>

Exemples :
  python main.py labyrinth Thomas
  python main.py shooter Agent_IA --agent Sniper
  python main.py reflex Bot --agent Réactif --data data/synthetic_sessions_500.csv
"""

import sys
import argparse
import os
import webbrowser

# ── Import des jeux ───────────────────────────────────────────────────────────
from games.reflex_game    import ReflexGame
import pygame
from games.reflex_game import ReflexGame
from games.labyrinth_game import LabyrinthGame
from games.shooter_game   import TwinStickShooter
from games.racing_game    import RacingGame

GAMES = {
    "reflex":    ReflexGame,
    "labyrinth": LabyrinthGame,
    "shooter":   TwinStickShooter,
    "racing":    RacingGame,
}

GAME_LABELS = {
    "reflex":    "🎯 Reflex",
    "labyrinth": "🌀 Labyrinth",
    "shooter":   "🚀 Shooter",
    "racing":    "🏎️  Racing",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="SISE Ultimate Games — Lance un jeu (humain ou agent IA)"
    )
    parser.add_argument(
        "game_id",
        choices=list(GAMES.keys()),
        help="Jeu à lancer : reflex | labyrinth | shooter | racing"
    )
    parser.add_argument(
        "player_name",
        help="Nom du joueur (ex: Thomas) ou de l'agent (ex: Agent_IA)"
    )
    parser.add_argument(
        "--agent",
        metavar="PROFILE_OR_PLAYER",
        default=None,
        help="Active le mode agent IA avec le profil ou joueur spécifié "
             "(ex: Sniper, modou…)"
    )
    parser.add_argument(
        "--mode",
        choices=["profile", "player"],
        default="profile",
        help="Mode d'imitation : 'profile' (CSV synthétique) ou 'player' (Supabase réel)"
    )
    parser.add_argument(
        "--data",
        metavar="CSV_PATH",
        default=None,
        help="Chemin vers le CSV de sessions (défaut: data/synthetic_sessions_500.csv)"
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="Liste les profils disponibles pour le jeu sélectionné et quitte"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Mode sans fenêtre (tests automatisés)"
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=1.0,
        help="Niveau de bruit du replay (0.0=exact, 1.0=défaut, 2.0=bruité)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Lister les profils ────────────────────────────────────────────────────
    if args.list_profiles:
        from core.agent import GameAgent
        profiles = GameAgent.list_profiles(args.game_id, args.data)
        print(f"\nProfils disponibles pour {args.game_id.upper()} :")
        for p in profiles:
            print(f"  → {p}")
        sys.exit(0)

    # ── Récupérer la classe du jeu ────────────────────────────────────────────
    GameClass = GAMES.get(args.game_id)
    if GameClass is None:
        print(f"❌ Jeu inconnu : {args.game_id}")
        print(f"   Jeux disponibles : {', '.join(GAMES.keys())}")
        sys.exit(1)

    label = GAME_LABELS.get(args.game_id, args.game_id)
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"  Joueur : {args.player_name}")

    # ── Mode agent ────────────────────────────────────────────────────────────
    agent = None
    if args.agent:
        from core.agent import GameAgent
        mode_label = "joueur réel Supabase" if args.mode == "player" else "profil synthétique"
        print(f"  Mode  : 🤖 Agent IA — '{args.agent}' ({mode_label})")
        print(f"{'='*50}\n")
        try:
            agent = GameAgent(
                game_id=args.game_id,
                profile_name=args.agent,
                data_path=args.data,
                mode=args.mode,
                noise_level=args.noise,
            )
        except Exception as e:
            print(f"❌ Erreur chargement agent : {e}")
            sys.exit(1)
    else:
        print(f"  Mode  : 👤 Joueur humain")
        print(f"{'='*50}\n")

    # ── Lancer le jeu ─────────────────────────────────────────────────────────
    try:
        game     = GameClass(
            player_name=args.player_name,
            headless=args.headless,
            agent=agent,
        )
        features = game.run()

        # ── Résumé post-session ───────────────────────────────────────────────
        print(f"\n{'='*50}")
        print(f"  Session terminée")
        print(f"  Score    : {getattr(features, 'score', '?')}")
        print(f"  Durée    : {getattr(features, 'duration_sec', 0):.1f}s")
        if agent:
            print(f"  Profil   : {agent.profile_name}")
            print(f"  Score attendu : ~{agent.profile.expected_score:.0f}")
        print(f"{'='*50}\n")

        # ── Résumé LLM Mistral (joueurs humains uniquement) ──────────────────
        if not agent:
            try:
                from core.llm_summary import generate_session_summary, save_summary_to_supabase
                from core.supabase_client import fetch_sessions_by_game, fetch_sessions_by_player

                all_sessions    = fetch_sessions_by_game(args.game_id)
                player_sessions = fetch_sessions_by_player(args.player_name)

                print("\n🧠 Génération du résumé IA (Mistral)...")
                summary = generate_session_summary(features, all_sessions, player_sessions)

                # ── Affichage terminal ────────────────────────────────────────
                sep = "═" * 52
                print(f"\n{sep}")
                emoji = summary.get("emoji_humeur", "🎮")
                titre = summary.get("titre", "Résumé de session")
                print(f"  {emoji}  {titre}")
                print(sep)

                print(f"\n{summary.get('resume', '')}\n")

                pf = summary.get("points_forts", [])
                if pf:
                    print("💪 Points forts :")
                    for point in pf:
                        print(f"   ✓ {point}")

                axes = summary.get("axes_amelioration", [])
                if axes:
                    print("\n📈 Axes d'amélioration :")
                    for axe in axes:
                        print(f"   → {axe}")

                cl_glob  = summary.get("classement_global", "")
                cl_perso = summary.get("classement_personnel", "")
                if cl_glob:
                    print(f"\n🏆 {cl_glob}")
                if cl_perso:
                    print(f"📊 {cl_perso}")

                conseil = summary.get("conseil", "")
                if conseil:
                    print(f"\n💡 Conseil : {conseil}")

                print(f"\n{sep}\n")

                # Sauvegarde Supabase (synchrone pour que le dashboard la trouve immédiatement)
                import time as _time
                ts = int(_time.time())
                save_summary_to_supabase(features, summary)

                # Ouvrir le navigateur sur la page post-session avec timestamp
                url = (f"http://127.0.0.1:8050/?player={args.player_name}"
                       f"&game={args.game_id}&ts={ts}")
                webbrowser.open(url)
                print(f"🌐 Résumé détaillé disponible sur : {url}")

            except Exception as e:
                print(f"⚠️  Résumé LLM indisponible : {e}")

    except KeyboardInterrupt:
        print("\n⏹  Interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()



# """
# main.py — Point d'entrée principal
# Lance un jeu en mode humain ou en mode agent IA.

# Usage :
#   # Mode humain
#   python main.py <game_id> <player_name>

#   # Mode agent
#   python main.py <game_id> <player_name> --agent <profile_name>
#   python main.py <game_id> <player_name> --agent <profile_name> --data <path_csv>

# Exemples :
#   python main.py labyrinth Thomas
#   python main.py shooter Agent_IA --agent Sniper
#   python main.py reflex Bot --agent Réactif --data data/synthetic_sessions_500.csv
# """

# import sys
# import argparse
# import os

# # ── Import des jeux ───────────────────────────────────────────────────────────
# from games.reflex_game    import ReflexGame
# from games.labyrinth_game import LabyrinthGame
# from games.shooter_game   import TwinStickShooter
# from games.racing_game    import RacingGame

# GAMES = {
#     "reflex":    ReflexGame,
#     "labyrinth": LabyrinthGame,
#     "shooter":   TwinStickShooter,
#     "racing":    RacingGame,
# }

# GAME_LABELS = {
#     "reflex":    "🎯 Reflex",
#     "labyrinth": "🌀 Labyrinth",
#     "shooter":   "🚀 Shooter",
#     "racing":    "🏎️  Racing",
# }


# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="SISE Ultimate Games — Lance un jeu (humain ou agent IA)"
#     )
#     parser.add_argument(
#         "game_id",
#         choices=list(GAMES.keys()),
#         help="Jeu à lancer : reflex | labyrinth | shooter | racing"
#     )
#     parser.add_argument(
#         "player_name",
#         help="Nom du joueur (ex: Thomas) ou de l'agent (ex: Agent_IA)"
#     )
#     parser.add_argument(
#         "--agent",
#         metavar="PROFILE_OR_PLAYER",
#         default=None,
#         help="Active le mode agent IA avec le profil ou joueur spécifié "
#              "(ex: Sniper, modou…)"
#     )
#     parser.add_argument(
#         "--mode",
#         choices=["profile", "player"],
#         default="profile",
#         help="Mode d'imitation : 'profile' (CSV synthétique) ou 'player' (Supabase réel)"
#     )
#     parser.add_argument(
#         "--data",
#         metavar="CSV_PATH",
#         default=None,
#         help="Chemin vers le CSV de sessions (défaut: data/synthetic_sessions_500.csv)"
#     )
#     parser.add_argument(
#         "--list-profiles",
#         action="store_true",
#         help="Liste les profils disponibles pour le jeu sélectionné et quitte"
#     )
#     parser.add_argument(
#         "--headless",
#         action="store_true",
#         help="Mode sans fenêtre (tests automatisés)"
#     )
#     return parser.parse_args()


# def main():
#     args = parse_args()

#     # ── Lister les profils ────────────────────────────────────────────────────
#     if args.list_profiles:
#         from core.agent import GameAgent
#         profiles = GameAgent.list_profiles(args.game_id, args.data)
#         print(f"\nProfils disponibles pour {args.game_id.upper()} :")
#         for p in profiles:
#             print(f"  → {p}")
#         sys.exit(0)

#     # ── Récupérer la classe du jeu ────────────────────────────────────────────
#     GameClass = GAMES.get(args.game_id)
#     if GameClass is None:
#         print(f"❌ Jeu inconnu : {args.game_id}")
#         print(f"   Jeux disponibles : {', '.join(GAMES.keys())}")
#         sys.exit(1)

#     label = GAME_LABELS.get(args.game_id, args.game_id)
#     print(f"\n{'='*50}")
#     print(f"  {label}")
#     print(f"  Joueur : {args.player_name}")

#     # ── Mode agent ────────────────────────────────────────────────────────────
#     agent = None
#     if args.agent:
#         from core.agent import GameAgent
#         mode_label = "joueur réel Supabase" if args.mode == "player" else "profil synthétique"
#         print(f"  Mode  : 🤖 Agent IA — '{args.agent}' ({mode_label})")
#         print(f"{'='*50}\n")
#         try:
#             agent = GameAgent(
#                 game_id=args.game_id,
#                 profile_name=args.agent,
#                 data_path=args.data,
#                 mode=args.mode,
#             )
#         except Exception as e:
#             print(f"❌ Erreur chargement agent : {e}")
#             sys.exit(1)
#     else:
#         print(f"  Mode  : 👤 Joueur humain")
#         print(f"{'='*50}\n")

#     # ── Lancer le jeu ─────────────────────────────────────────────────────────
#     try:
#         game     = GameClass(
#             player_name=args.player_name,
#             headless=args.headless,
#             agent=agent,
#         )
#         features = game.run()

#         # ── Résumé post-session ───────────────────────────────────────────────
#         print(f"\n{'='*50}")
#         print(f"  Session terminée")
#         print(f"  Score    : {getattr(features, 'score', '?')}")
#         print(f"  Durée    : {getattr(features, 'duration_sec', 0):.1f}s")
#         if agent:
#             print(f"  Profil   : {agent.profile_name}")
#             print(f"  Score attendu : ~{agent.profile.expected_score:.0f}")
#         print(f"{'='*50}\n")

#         # ── Résumé LLM (joueurs humains uniquement) ───────────────────────────
#         if not agent:
#             try:
#                 from core.llm_summary import generate_session_summary, save_summary_to_supabase
#                 from core.supabase_client import fetch_sessions_by_game, fetch_sessions_by_player

#                 print("🧠 Génération du résumé IA...")
#                 all_sessions    = fetch_sessions_by_game(args.game_id)
#                 player_sessions = fetch_sessions_by_player(args.player_name)

#                 summary = generate_session_summary(
#                     features,
#                     all_sessions=all_sessions,
#                     player_sessions=player_sessions,
#                 )

#                 # Affichage terminal
#                 print(f"\n{'='*50}")
#                 print(f"  {summary.get('emoji_humeur','')} {summary.get('titre','Résumé session')}")
#                 print(f"{'='*50}")
#                 print(f"\n{summary.get('resume','')}")
#                 print(f"\n💪 Points forts :")
#                 for pf in summary.get("points_forts", []):
#                     print(f"   → {pf}")
#                 print(f"\n📈 Axes d'amélioration :")
#                 for ax in summary.get("axes_amelioration", []):
#                     print(f"   → {ax}")
#                 print(f"\n🏆 {summary.get('classement_global','')}")
#                 print(f"📊 {summary.get('classement_personnel','')}")
#                 print(f"\n💡 Conseil : {summary.get('conseil','')}")
#                 print(f"{'='*50}\n")

#                 # Sauvegarde Supabase
#                 save_summary_to_supabase(features, summary)

#             except Exception as e:
#                 print(f"⚠️  Résumé LLM indisponible : {e}")

#     except KeyboardInterrupt:
#         print("\n⏹  Interrompu par l'utilisateur")
#     except Exception as e:
#         print(f"\n❌ Erreur : {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)


# if __name__ == "__main__":
#     main()

