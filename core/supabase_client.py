"""
supabase_client.py — Connexion Supabase et insertion des sessions
À importer dans base_game.py uniquement.

Usage :
    from core.supabase_client import save_features_to_supabase
    save_features_to_supabase(features)
"""

import os
from dataclasses import asdict
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Colonnes présentes dans la table Supabase `sessions`
# (on exclut `source` qui n'est pas dans le schéma DB)
SESSION_COLUMNS = [
    "player_name",
    "game_id",
    "duration_sec",
    "btn_press_rate",
    "btn_variety",
    "btn_hold_avg_ms",
    "lx_mean",
    "ly_mean",
    "lx_std",
    "ly_std",
    "lx_direction_changes",
    "rx_mean",
    "ry_mean",
    "rx_std",
    "ry_std",
    "lt_mean",
    "rt_mean",
    "lt_brutality",
    "rt_brutality",
    "reaction_time_avg_ms",
    "input_regularity",
    "score",
]


def _get_client():
    """Crée et retourne le client Supabase (lazy init)"""
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise EnvironmentError(
            "❌ SUPABASE_URL ou SUPABASE_KEY manquant dans le fichier .env"
        )
    from supabase import create_client
    return create_client(SUPABASE_URL, SUPABASE_KEY)


def save_features_to_supabase(features) -> bool:
    """
    Insère une SessionFeatures dans la table `sessions` de Supabase.

    Args:
        features : objet SessionFeatures retourné par recorder.stop()

    Returns:
        True si succès, False si erreur (l'erreur est loggée mais pas levée
        pour ne pas bloquer le jeu)
    """
    try:
        client = _get_client()

        # Convertir le dataclass en dict et filtrer les colonnes connues
        row = asdict(features)
        filtered_row = {k: v for k, v in row.items() if k in SESSION_COLUMNS}

        result = client.table("sessions").insert(filtered_row).execute()

        inserted_id = result.data[0]["id"] if result.data else "?"
        print(f"☁️  Session envoyée sur Supabase — id: {inserted_id}")
        return True

    except Exception as e:
        print(f"⚠️  Supabase insert échoué (session conservée en CSV) : {e}")
        return False


def fetch_all_sessions() -> list[dict]:
    """
    Récupère toutes les sessions depuis Supabase.
    Utilisé par le pipeline ML et le dashboard Dash.

    Returns:
        Liste de dicts (une entrée par session)
    """
    try:
        client = _get_client()
        result = client.table("sessions").select("*").order("created_at").execute()
        return result.data or []
    except Exception as e:
        print(f"⚠️  Supabase fetch échoué : {e}")
        return []


def fetch_sessions_by_player(player_name: str) -> list[dict]:
    """Récupère toutes les sessions d'un joueur spécifique"""
    try:
        client = _get_client()
        result = (
            client.table("sessions")
            .select("*")
            .eq("player_name", player_name)
            .order("created_at")
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"⚠️  Supabase fetch player échoué : {e}")
        return []


def fetch_sessions_by_game(game_id: str) -> list[dict]:
    """Récupère toutes les sessions d'un jeu spécifique"""
    try:
        client = _get_client()
        result = (
            client.table("sessions")
            .select("*")
            .eq("game_id", game_id)
            .order("created_at")
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"⚠️  Supabase fetch game échoué : {e}")
        return []


def fetch_latest_sessions(limit: int = 20) -> list[dict]:
    """
    Récupère les N dernières sessions.
    Utilisé par le dashboard Dash pour le live feed.
    """
    try:
        client = _get_client()
        result = (
            client.table("sessions")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data or []
    except Exception as e:
        print(f"⚠️  Supabase fetch latest échoué : {e}")
        return []


def save_profile_to_supabase(player_name: str, cluster_id: int,
                              cluster_name: str, features_dict: dict) -> bool:
    """
    Upsert un profil ML dans la table `profils_ml`.
    Appelé par le pipeline de clustering après calcul.

    Args:
        player_name  : nom du joueur
        cluster_id   : numéro du cluster (0, 1, 2, 3)
        cluster_name : label du cluster ('Agressif', 'Prudent', etc.)
        features_dict: dict des features moyennes du joueur
    """
    try:
        import json
        client = _get_client()
        row = {
            "player_name": player_name,
            "cluster_id": cluster_id,
            "cluster_name": cluster_name,
            "features_json": features_dict,
        }
        # Upsert : met à jour si le joueur existe déjà
        result = (
            client.table("profils_ml")
            .upsert(row, on_conflict="player_name")
            .execute()
        )
        print(f"🧬 Profil '{cluster_name}' enregistré pour {player_name}")
        return True
    except Exception as e:
        print(f"⚠️  Supabase upsert profil échoué : {e}")
        return False


def fetch_all_profiles() -> list[dict]:
    """Récupère tous les profils ML calculés"""
    try:
        client = _get_client()
        result = client.table("profils_ml").select("*").execute()
        return result.data or []
    except Exception as e:
        print(f"⚠️  Supabase fetch profils échoué : {e}")
        return []

def send_inputs_batch(batch: list[dict]) -> bool:
    """
    Insère un batch d'inputs live dans Supabase.
    Appelé toutes les 500ms depuis la boucle pygame.
    """
    try:
        client = _get_client()
        client.table("inputs_live").insert(batch).execute()
        return True
    except Exception as e:
        print(f"⚠️ inputs_live batch échoué : {e}")
        return False
    
def fetch_live_inputs(session_token: str = None, limit: int = 60) -> list[dict]:
    """
    Récupère les derniers inputs live.
    Si session_token fourni, filtre sur la session en cours.
    """
    try:
        client = _get_client()
        query = client.table("inputs_live").select("*").order("captured_at", desc=True).limit(limit)
        if session_token:
            query = query.eq("session_token", session_token)
        result = query.execute()
        return list(reversed(result.data or []))
    except Exception as e:
        print(f"⚠️ fetch inputs_live échoué : {e}")
        return []