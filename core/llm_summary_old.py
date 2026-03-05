"""
llm_summary.py — Résumé LLM post-session via Mistral API
Retry automatique sur 429, timeout 10s max, non-bloquant.
"""

import os, json, time, threading, requests

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip().strip('"')
MODEL   = "mistral-small-latest"
API_URL = "https://api.mistral.ai/v1/chat/completions"
TIMEOUT = (5, 15)  # (connect, read) — plus fiable sur Windows
MAX_RETRIES = 3
RETRY_DELAY = 5    # secondes entre retries sur 429
MAX_WAIT    = 30   # attente max par retry — si Retry-After > MAX_WAIT, on abandonne

GAME_LABELS = {
    "reflex":    "Reflex Challenge",
    "labyrinth": "Labyrinthe",
    "shooter":   "Shooter",
    "racing":    "Racing",
}


def _build_prompt(features, all_sessions, player_sessions):
    game_label = GAME_LABELS.get(features.game_id, features.game_id)
    n_total    = len(all_sessions)
    n_player   = len(player_sessions)

    if all_sessions:
        scores_sorted = sorted([s.get("score", 0) for s in all_sessions], reverse=True)
        rank_global   = next((i+1 for i, s in enumerate(scores_sorted) if s <= features.score), n_total)
        pct_global    = round((1 - rank_global / max(n_total, 1)) * 100, 1)
    else:
        rank_global, pct_global = "?", "?"

    if len(player_sessions) > 1:
        p_scores      = sorted([s.get("score", 0) for s in player_sessions], reverse=True)
        rank_personal = next((i+1 for i, s in enumerate(p_scores) if s <= features.score), n_player)
        personal_best = max(s.get("score", 0) for s in player_sessions)
        personal_avg  = round(sum(s.get("score", 0) for s in player_sessions) / n_player, 0)
    else:
        rank_personal, personal_best, personal_avg = 1, features.score, features.score

    global_avg  = round(sum(s.get("score", 0) for s in all_sessions) / max(n_total, 1), 0) if all_sessions else 0
    global_best = max((s.get("score", 0) for s in all_sessions), default=0)

    return f"""Tu es un coach gaming pour le projet SISE Gaming Analytics.
Génère un résumé de session personnalisé en français.

Session : {features.player_name} | {game_label} | Score: {features.score} | Durée: {features.duration_sec:.0f}s
Boutons: {features.btn_press_rate:.2f}/s | Joystick: lx={features.lx_std:.3f} ly={features.ly_std:.3f} | Régularité: {features.input_regularity:.2f}
Rang global: {rank_global}/{n_total} ({pct_global}% battus) | Rang perso: {rank_personal}/{n_player}
Meilleur perso: {personal_best} | Moyenne perso: {personal_avg} | Moyenne globale: {global_avg} | Record: {global_best}

Réponds UNIQUEMENT avec du JSON valide sans markdown:
{{"titre":"titre accrocheur","resume":"2-3 phrases personnalisées avec les vraies stats","points_forts":["point précis 1","point précis 2"],"axes_amelioration":["axe précis 1","axe précis 2"],"classement_global":"phrase avec les chiffres réels","classement_personnel":"phrase sur évolution perso","conseil":"conseil précis et actionnable","emoji_humeur":"emoji"}}"""


def generate_session_summary(features, all_sessions=None, player_sessions=None):
    """Appelle Mistral avec retry sur 429. Retourne mock si échec total."""
    if not MISTRAL_API_KEY:
        print("⚠️  MISTRAL_API_KEY manquant — mock")
        return _mock_summary(features)

    prompt = _build_prompt(features, all_sessions or [], player_sessions or [])

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                API_URL,
                headers={
                    "Content-Type":  "application/json",
                    "Authorization": f"Bearer {MISTRAL_API_KEY}",
                },
                json={"model": MODEL, "messages": [{"role": "user", "content": prompt}]},
                timeout=TIMEOUT,
            )

            if response.status_code == 429:
                raw_ra = response.headers.get("Retry-After")
                if raw_ra is None:
                    # Pas de Retry-After → quota épuisé (daily/monthly), inutile de retry
                    print("⏳ Mistral 429 — quota épuisé (pas de Retry-After) → mock immédiat")
                    break
                try:
                    retry_after = int(float(raw_ra))
                except (ValueError, TypeError):
                    retry_after = RETRY_DELAY * attempt
                if retry_after > MAX_WAIT:
                    print(f"⏳ Mistral 429 — Retry-After={retry_after}s > max({MAX_WAIT}s) → mock immédiat")
                    break
                print(f"⏳ Mistral 429 — attente {retry_after}s (tentative {attempt}/{MAX_RETRIES})")
                time.sleep(retry_after)
                continue

            # Log explicite pour 401/403/500 avant raise_for_status
            if not response.ok:
                print(f"⚠️  HTTP {response.status_code} Mistral : {response.text[:200]}")
            response.raise_for_status()

            raw = response.json()["choices"][0]["message"]["content"].strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            # Extraction robuste : isole le bloc JSON si du texte l'entoure
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end != -1:
                raw = raw[start : end + 1]
            result = json.loads(raw)
            print(f"✅ Résumé IA généré pour {features.player_name} @ {features.game_id}")
            return result

        except requests.exceptions.Timeout:
            print(f"⚠️  Mistral timeout (tentative {attempt}/{MAX_RETRIES})")
            if attempt < MAX_RETRIES:
                time.sleep(2)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON invalide : {e}")
            return _mock_summary(features)
        except Exception as e:
            print(f"⚠️  Erreur API : {e}")
            if attempt < MAX_RETRIES:
                time.sleep(2)

    print("⚠️  Toutes les tentatives ont échoué — mock utilisé")
    return _mock_summary(features)


def generate_and_save_async(features, all_sessions=None, player_sessions=None):
    """Sauvegarde un mock immédiatement, puis tente le vrai LLM en background."""
    # Mock sauvegardé de suite → dashboard l'affiche en quelques secondes
    mock = _mock_summary(features)
    save_summary_to_supabase(features, mock)
    print("🧠 Résumé mock sauvegardé — LLM en cours en background...")

    def _run():
        try:
            summary = generate_session_summary(features, all_sessions, player_sessions)
            if not summary.get("mock"):
                # Résumé LLM réel → insert une seconde entrée plus récente
                save_summary_to_supabase(features, summary)
                print("✅ Résumé LLM remplacé dans Supabase")
        except Exception as e:
            # Ne pas laisser le thread mourir silencieusement
            print(f"⚠️  Thread LLM — exception non gérée : {e}")

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t


def _mock_summary(features):
    game_label = GAME_LABELS.get(features.game_id, features.game_id)
    return {
        "titre":                f"Session de {features.player_name} — {game_label}",
        "resume":               f"{features.player_name} a joué au {game_label} pendant "
                                f"{features.duration_sec:.0f}s avec un score de {features.score}.",
        "points_forts":         ["Session complétée", "Données enregistrées"],
        "axes_amelioration":    ["Continue à jouer pour affiner ton profil"],
        "classement_global":    "Classement en cours de calcul...",
        "classement_personnel": "Première session ou données insuffisantes",
        "conseil":              "Lance une nouvelle partie pour améliorer ton profil !",
        "emoji_humeur":         "🎮",
        "mock":                 True,
    }


def chat_with_llm(message: str, history: list, context: str = "") -> str:
    """
    Conversation multi-tour avec Mistral.
    history : liste de {"role": "user"|"assistant", "content": str}
    context : données sessions injectées dans le system prompt
    Retourne la réponse texte de l'assistant.
    """
    if not MISTRAL_API_KEY:
        return "Service IA indisponible (clé API manquante). Configure MISTRAL_API_KEY dans le .env"

    system_prompt = f"""Tu es un coach gaming expert pour le projet SISE Gaming Analytics.
Tu analyses les performances des joueurs sur 4 jeux : Reflex, Labyrinthe, Shooter, Racing.
Tu peux répondre à des questions sur les classements, comparaisons, conseils et analyses de style de jeu.
Réponds en français, de manière concise et engageante. Utilise des données précises quand disponibles.

Contexte — données sessions disponibles :
{context[:3000] if context else "Aucune donnée chargée."}"""

    messages = [{"role": "system", "content": system_prompt}]
    for m in (history or [])[-10:]:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": message})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {MISTRAL_API_KEY}",
                         "Content-Type": "application/json"},
                json={"model": MODEL, "messages": messages},
                timeout=TIMEOUT,  # utilise la constante globale (5, 15)
            )
            if r.status_code == 429:
                try:
                    wait = min(int(float(r.headers.get("Retry-After", RETRY_DELAY * attempt))), MAX_WAIT)
                except (ValueError, TypeError):
                    wait = RETRY_DELAY * attempt
                if wait > MAX_WAIT:
                    break
                time.sleep(wait)
                continue
            if not r.ok:
                print(f"⚠️  Chat HTTP {r.status_code} : {r.text[:150]}")
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES:
                time.sleep(2)
        except Exception as e:
            return f"Erreur LLM : {str(e)[:100]}"

    return "Délai de réponse dépassé. Réessaie dans quelques secondes."


def save_summary_to_supabase(features, summary):
    try:
        from core.supabase_client import _get_client
        client = _get_client()
        row = {
            "player_name":  features.player_name,
            "game_id":      features.game_id,
            "score":        float(features.score),
            "duration_sec": float(features.duration_sec),
            "session_data": {
                "btn_press_rate":   features.btn_press_rate,
                "btn_variety":      features.btn_variety,
                "lx_std":           features.lx_std,
                "ly_std":           features.ly_std,
                "input_regularity": features.input_regularity,
            },
            "summary_md": json.dumps(summary, ensure_ascii=False),
        }
        client.table("summaries").insert(row).execute()
        print(f"☁️  Résumé sauvegardé — {features.player_name} score={features.score}")
        return True
    except Exception as e:
        print(f"⚠️  Save summary échoué : {e}")
        return False


def fetch_latest_summaries(limit=10):
    try:
        from core.supabase_client import _get_client
        client = _get_client()
        result = (
            client.table("summaries")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        rows = result.data or []
        for row in rows:
            raw = row.get("summary_md") or "{}"
            try:
                row["summary_json"] = json.loads(raw)
            except Exception:
                row["summary_json"] = {}
        return rows
    except Exception as e:
        print(f"⚠️  fetch summaries échoué : {e}")
        return []

# """
# llm_summary.py — Résumé LLM post-session via Mistral API
# Table Supabase `summaries` :
#   player_name TEXT, game_id TEXT, score FLOAT, duration_sec FLOAT,
#   session_data JSONB, summary_md TEXT, created_at TIMESTAMPTZ
# """

# import os
# import json
# import requests

# MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip().strip('"')
# MISTRAL_MODEL   = "mistral-small-latest"
# API_URL         = "https://api.mistral.ai/v1/chat/completions"

# GAME_LABELS = {
#     "reflex":    "Reflex Challenge",
#     "labyrinth": "Labyrinthe",
#     "shooter":   "Shooter",
#     "racing":    "Racing",
# }


# def _build_prompt(features, all_sessions, player_sessions):
#     game_label = GAME_LABELS.get(features.game_id, features.game_id)
#     n_total    = len(all_sessions)
#     n_player   = len(player_sessions)

#     if all_sessions:
#         scores_sorted = sorted([s.get("score", 0) for s in all_sessions], reverse=True)
#         rank_global   = next((i+1 for i, s in enumerate(scores_sorted) if s <= features.score), n_total)
#         pct_global    = round((1 - rank_global / max(n_total, 1)) * 100, 1)
#     else:
#         rank_global, pct_global = "?", "?"

#     if len(player_sessions) > 1:
#         p_scores      = sorted([s.get("score", 0) for s in player_sessions], reverse=True)
#         rank_personal = next((i+1 for i, s in enumerate(p_scores) if s <= features.score), n_player)
#         personal_best = max(s.get("score", 0) for s in player_sessions)
#         personal_avg  = round(sum(s.get("score", 0) for s in player_sessions) / n_player, 0)
#     else:
#         rank_personal = 1
#         personal_best = features.score
#         personal_avg  = features.score

#     global_avg  = round(sum(s.get("score", 0) for s in all_sessions) / max(n_total, 1), 0) if all_sessions else 0
#     global_best = max((s.get("score", 0) for s in all_sessions), default=0)

#     return f"""Tu es un coach gaming expert pour le projet SISE Gaming Analytics.
# Genere un resume de session engageant et personnalise en francais.

# Session :
# - Joueur : {features.player_name}
# - Jeu : {game_label}
# - Score : {features.score}
# - Duree : {features.duration_sec:.1f}s
# - Frequence boutons : {features.btn_press_rate:.2f}/s
# - Agitation joystick : lx={features.lx_std:.3f} ly={features.ly_std:.3f}
# - Regularite : {features.input_regularity:.2f}

# Classements :
# - Rang global : {rank_global}/{n_total} ({pct_global}% battus)
# - Rang personnel : {rank_personal}/{n_player}
# - Meilleur perso : {personal_best} | Moyenne perso : {personal_avg}
# - Moyenne globale : {global_avg} | Record absolu : {global_best}

# Reponds UNIQUEMENT avec du JSON valide sans markdown sans texte avant ou apres :
# {{
#   "titre": "titre accrocheur",
#   "resume": "2-3 phrases engageantes et personnalisees",
#   "points_forts": ["point 1", "point 2"],
#   "axes_amelioration": ["axe 1", "axe 2"],
#   "classement_global": "phrase sur le rang global",
#   "classement_personnel": "phrase sur evolution personnelle",
#   "conseil": "1 conseil precis pour la prochaine partie",
#   "emoji_humeur": "emoji"
# }}"""


# def generate_session_summary(features, all_sessions=None, player_sessions=None):
#     if not MISTRAL_API_KEY:
#         print("WARNING MISTRAL_API_KEY manquant - resume mock utilise")
#         return _mock_summary(features)

#     prompt = _build_prompt(features, all_sessions or [], player_sessions or [])

#     try:
#         response = requests.post(
#             API_URL,
#             headers={
#                 "Content-Type":  "application/json",
#                 "Authorization": f"Bearer {MISTRAL_API_KEY}",
#             },
#             json={
#                 "model":    MISTRAL_MODEL,
#                 "messages": [{"role": "user", "content": prompt}],
#             },
#             timeout=30,
#         )
#         response.raise_for_status()
#         raw = response.json()["choices"][0]["message"]["content"].strip()
#         raw = raw.replace("```json", "").replace("```", "").strip()
#         result = json.loads(raw)
#         print(f"Resume Mistral genere pour {features.player_name} @ {features.game_id}")
#         return result

#     except requests.exceptions.Timeout:
#         print("WARNING Mistral timeout - mock")
#         return _mock_summary(features)
#     except json.JSONDecodeError as e:
#         print(f"WARNING JSON invalide : {e}")
#         return _mock_summary(features)
#     except Exception as e:
#         print(f"WARNING Erreur API LLM : {e}")
#         return _mock_summary(features)


# def _mock_summary(features):
#     game_label = GAME_LABELS.get(features.game_id, features.game_id)
#     return {
#         "titre":                f"Session de {features.player_name} - {game_label}",
#         "resume":               f"{features.player_name} a joue au {game_label} pendant "
#                                 f"{features.duration_sec:.0f}s avec un score de {features.score}.",
#         "points_forts":         ["Session completee", "Donnees enregistrees"],
#         "axes_amelioration":    ["Continue a jouer pour affiner ton profil"],
#         "classement_global":    "Classement en cours de calcul...",
#         "classement_personnel": "Premiere session ou donnees insuffisantes",
#         "conseil":              "Lance une nouvelle partie pour ameliorer ton profil !",
#         "emoji_humeur":         "🎮",
#         "mock":                 True,
#     }


# def save_summary_to_supabase(features, summary):
#     """
#     Colonnes reelles de summaries :
#     player_name, game_id, score, duration_sec, session_data (JSONB), summary_md (TEXT)
#     """
#     try:
#         from core.supabase_client import _get_client
#         client = _get_client()
#         row = {
#             "player_name":  features.player_name,
#             "game_id":      features.game_id,
#             "score":        float(features.score),
#             "duration_sec": float(features.duration_sec),
#             "session_data": {
#                 "btn_press_rate":   features.btn_press_rate,
#                 "btn_variety":      features.btn_variety,
#                 "lx_std":           features.lx_std,
#                 "ly_std":           features.ly_std,
#                 "input_regularity": features.input_regularity,
#             },
#             "summary_md": json.dumps(summary, ensure_ascii=False),
#         }
#         client.table("summaries").insert(row).execute()
#         print(f"Resume sauvegarde pour {features.player_name} (score={features.score})")
#         return True
#     except Exception as e:
#         print(f"WARNING Save summary echoue : {e}")
#         return False


# def fetch_latest_summaries(limit=10):
#     try:
#         from core.supabase_client import _get_client
#         client = _get_client()
#         result = (
#             client.table("summaries")
#             .select("*")
#             .order("created_at", desc=True)
#             .limit(limit)
#             .execute()
#         )
#         rows = result.data or []
#         for row in rows:
#             raw = row.get("summary_md") or "{}"
#             try:
#                 row["summary_json"] = json.loads(raw)
#             except Exception:
#                 row["summary_json"] = {}
#         return rows
#     except Exception as e:
#         print(f"WARNING fetch summaries echoue : {e}")
#         return []