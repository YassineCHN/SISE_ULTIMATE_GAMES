"""
llm_summary.py — Résumé LLM post-session via Mistral API
Retry automatique sur 429, timeout 10s max, non-bloquant.
"""

import os, json, time, threading, requests
from dotenv import load_dotenv

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY", "").strip().strip('"')
MODEL   = "mistral-small-latest"
API_URL = "https://api.mistral.ai/v1/chat/completions"
TIMEOUT = 25    # secondes max par tentative
MAX_RETRIES = 3
RETRY_DELAY = 5 # secondes entre retries sur 429

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
        progression   = round(features.score - personal_avg, 0)
    else:
        rank_personal, personal_best, personal_avg, progression = 1, features.score, features.score, 0

    global_avg  = round(sum(s.get("score", 0) for s in all_sessions) / max(n_total, 1), 0) if all_sessions else 0
    global_best = max((s.get("score", 0) for s in all_sessions), default=0)

    # Interprétation des métriques pour guider l'IA
    btn_interp = (
        "très peu de boutons (joueur calme/stratège)" if features.btn_press_rate < 1 else
        "rythme d'appuis modéré (joueur équilibré)" if features.btn_press_rate < 2.5 else
        "fréquence élevée (joueur frénétique/agressif)" if features.btn_press_rate < 4 else
        "fréquence extrême (joueur ultra-réactif)"
    )
    joystick_interp = (
        "mouvements très précis et maîtrisés" if features.lx_std < 0.15 else
        "mouvements fluides et contrôlés" if features.lx_std < 0.30 else
        "mouvements amples et dynamiques" if features.lx_std < 0.45 else
        "mouvements chaotiques/imprévisibles"
    )
    regularity_interp = (
        "rythme très régulier et méthodique" if features.input_regularity < 0.1 else
        "rythme assez constant" if features.input_regularity < 0.25 else
        "jeu saccadé avec des accélérations" if features.input_regularity < 0.5 else
        "timing très irrégulier/impulsif"
    )
    vs_avg = "au-dessus de la moyenne" if features.score > global_avg else "en-dessous de la moyenne"
    prog_txt = f"+{progression:.0f} au-dessus de ta moyenne" if progression >= 0 else f"{progression:.0f} par rapport à ta moyenne"

    return f"""Tu es un coach gaming expert pour le projet SISE Gaming Analytics.
Génère une analyse de session COMPLÈTE, DÉTAILLÉE et COHÉRENTE en français.
Utilise OBLIGATOIREMENT les vraies valeurs numériques fournies dans tes phrases.
Ne génère pas de texte générique — chaque phrase doit être spécifique à cette session.

=== SESSION ===
Joueur : {features.player_name}
Jeu    : {game_label}
Score  : {features.score} pts  ({vs_avg} de {global_avg} pts)
Durée  : {features.duration_sec:.0f} secondes

=== MÉTRIQUES DÉTAILLÉES ===
• Fréquence boutons : {features.btn_press_rate:.2f} appuis/s → {btn_interp}
• Variété des boutons utilisés : {features.btn_variety:.0%}
• Agitation joystick G : lx={features.lx_std:.3f} | ly={features.ly_std:.3f} → {joystick_interp}
• Régularité des inputs : {features.input_regularity:.3f} → {regularity_interp}
• Brutalité gâchette G : {features.lt_brutality:.3f} | D : {features.rt_brutality:.3f}

=== CLASSEMENTS ===
• Rang global  : {rank_global} / {n_total} joueurs — {pct_global}% battus
• Rang perso   : {rank_personal} / {n_player} de tes sessions
• Ton record   : {personal_best} pts | Ta moyenne : {personal_avg} pts | {prog_txt}
• Record absolu: {global_best} pts | Moyenne globale : {global_avg} pts

Réponds UNIQUEMENT avec ce JSON valide (sans markdown, sans texte avant ou après) :
{{
  "titre": "titre accrocheur et personnalisé mentionnant {features.player_name} et/ou le jeu (max 65 chars)",
  "emoji_humeur": "1 emoji représentant parfaitement le niveau de la session",
  "resume": "3-4 phrases complètes incluant le score exact ({features.score}), la durée ({features.duration_sec:.0f}s) et une comparaison avec la moyenne ({global_avg}). Sois précis et engageant.",
  "analyse_style": "2-3 phrases analysant précisément le style de jeu basé sur les métriques : fréquence {features.btn_press_rate:.2f}/s signifie quoi pour ce joueur, joystick {features.lx_std:.3f} révèle quoi, régularité {features.input_regularity:.3f} indique quoi.",
  "profil_joueur": "label court et précis du profil de jeu (ex: Sniper Méthodique, Assaillant Frénétique, Stratège Posé, Explorateur Chaotique...)",
  "points_forts": [
    "point fort #1 basé sur une métrique précise avec chiffres",
    "point fort #2 basé sur une autre métrique",
    "point fort #3 si pertinent"
  ],
  "axes_amelioration": [
    "axe #1 très précis avec la métrique concernée et comment l'améliorer",
    "axe #2 concret et actionnable"
  ],
  "classement_global": "phrase précise : rang {rank_global}/{n_total}, {pct_global}% battus, comparaison avec record {global_best}",
  "classement_personnel": "phrase sur l'évolution : {prog_txt}, par rapport à {n_player} sessions",
  "conseil": "1 conseil TRÈS PRÉCIS et actionnable pour la prochaine partie, basé sur la métrique la plus faible",
  "objectif": "1 objectif chiffré concret pour la prochaine partie (ex: dépasser X pts, réduire la fréquence boutons à Y/s...)"
}}"""


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
                wait = RETRY_DELAY * attempt
                print(f"⏳ Mistral rate limit — attente {wait}s (tentative {attempt}/{MAX_RETRIES})")
                time.sleep(wait)
                continue

            response.raise_for_status()
            raw    = response.json()["choices"][0]["message"]["content"].strip()
            raw    = raw.replace("```json", "").replace("```", "").strip()
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
    """Lance génération + save en background. Non-bloquant."""
    def _run():
        summary = generate_session_summary(features, all_sessions, player_sessions)
        save_summary_to_supabase(features, summary)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    print("🧠 Résumé IA en cours (visible dans le dashboard onglet Résumés sous ~30s)...")
    return t


def _mock_summary(features):
    game_label = GAME_LABELS.get(features.game_id, features.game_id)
    btn_label = (
        "très calme et stratège" if features.btn_press_rate < 1 else
        "équilibré" if features.btn_press_rate < 2.5 else
        "agressif et frénétique"
    )
    joy_label = (
        "précis et contrôlés" if features.lx_std < 0.25 else
        "amples et dynamiques" if features.lx_std < 0.45 else "imprévisibles"
    )
    return {
        "titre":                f"{features.player_name} — Session {game_label} ({features.score} pts)",
        "emoji_humeur":         "🎮",
        "resume":               (
            f"{features.player_name} a joué au {game_label} pendant {features.duration_sec:.0f}s "
            f"et a obtenu un score de {features.score} points. "
            f"Avec {features.btn_press_rate:.2f} appuis/s, le style de jeu est {btn_label}. "
            f"Les mouvements de joystick (lx={features.lx_std:.3f}) sont {joy_label}."
        ),
        "analyse_style":        (
            f"La fréquence de {features.btn_press_rate:.2f} appuis/s révèle un joueur {btn_label}. "
            f"La régularité des inputs à {features.input_regularity:.3f} indique un rythme "
            f"{'méthodique et constant' if features.input_regularity < 0.25 else 'variable avec des phases impulsives'}. "
            f"Les mouvements de joystick {joy_label} suggèrent une approche "
            f"{'précise et calculée' if features.lx_std < 0.3 else 'dynamique et réactive'}."
        ),
        "profil_joueur":        (
            "Stratège Posé" if features.btn_press_rate < 1 and features.lx_std < 0.25 else
            "Tireur de Précision" if features.lx_std < 0.2 else
            "Assaillant Frénétique" if features.btn_press_rate > 3 else
            "Joueur Équilibré"
        ),
        "points_forts":         [
            f"Score de {features.score} pts enregistré avec succès",
            f"Session complète de {features.duration_sec:.0f}s ({features.btn_press_rate:.2f} inputs/s)",
            f"Données de session sauvegardées pour le suivi de progression",
        ],
        "axes_amelioration":    [
            f"Travailler la régularité des inputs (actuelle : {features.input_regularity:.3f})",
            f"Optimiser l'utilisation du joystick pour plus de précision (lx_std={features.lx_std:.3f})",
        ],
        "classement_global":    "Classement global en cours de calcul (API IA indisponible).",
        "classement_personnel": f"Session enregistrée — données insuffisantes pour calculer l'évolution personnelle.",
        "conseil":              (
            f"Concentre-toi sur la régularité de tes inputs pour maintenir un rythme constant. "
            f"Vise un écart-type inférieur à 0.15 (actuellement {features.input_regularity:.3f})."
        ),
        "objectif":             f"Dépasser {features.score + max(100, features.score // 10)} pts à la prochaine partie.",
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
                timeout=15,
            )
            if r.status_code == 429:
                time.sleep(RETRY_DELAY * attempt)
                continue
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