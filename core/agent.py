"""
agent.py — Agent imitateur IA
Charge le profil comportemental d'un joueur/profil depuis les données
(synthétiques ou réelles) et génère des ControllerState frame par frame
qui reproduisent fidèlement ce style de jeu.
"""

import numpy as np
import pandas as pd
import time
import os
from dataclasses import dataclass, field
from typing import Optional

# Import ControllerState sans dépendre de pygame au chargement
# (pygame est initialisé par BaseGame avant que l'agent soit utilisé)
from core.controller import ControllerState


# ─────────────────────────────────────────────────────────────────────────────
# PROFIL COMPORTEMENTAL
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BehaviorProfile:
    """Statistiques comportementales d'un profil pour un jeu donné."""
    game_id:        str
    profile_name:   str
    n_sessions:     int

    # Joystick gauche
    lx_mean:  float = 0.0
    lx_std:   float = 0.3
    ly_mean:  float = 0.0
    ly_std:   float = 0.3

    # Joystick droit
    rx_std:   float = 0.0
    ry_std:   float = 0.0

    # Gâchettes
    lt_mean:  float = 0.0
    lt_std:   float = 0.0
    rt_mean:  float = 0.0
    rt_std:   float = 0.0

    # Boutons
    btn_press_rate:   float = 0.1   # probabilité d'appui par frame
    btn_hold_avg_ms:  float = 200.0 # durée moyenne d'appui en ms
    btn_hold_std_ms:  float = 100.0
    btn_variety:      float = 0.2   # diversité boutons utilisés

    # Score / durée attendus
    expected_score:   float = 1000.0
    expected_duration: float = 60.0

    # Régularité (0 = erratique, élevé = très régulier)
    input_regularity: float = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES PROFILS
# ─────────────────────────────────────────────────────────────────────────────

# Colonnes features disponibles dans le CSV
FEATURE_COLS = [
    "btn_press_rate", "btn_variety", "btn_hold_avg_ms",
    "lx_mean", "ly_mean", "lx_std", "ly_std", "lx_direction_changes",
    "rx_mean", "ry_mean", "rx_std", "ry_std",
    "lt_mean", "rt_mean", "lt_brutality", "rt_brutality",
    "input_regularity", "score", "duration_sec",
]

# Chemin vers les données (cherche dans plusieurs endroits)
def _find_data_file():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "data", "synthetic_sessions_500.csv"),
        os.path.join(os.path.dirname(__file__), "..", "synthetic_sessions_500.csv"),
        "data/synthetic_sessions_500.csv",
        "synthetic_sessions_500.csv",
    ]
    for path in candidates:
        if os.path.exists(path):
            return os.path.abspath(path)
    return None


def _build_profile_from_group(group: pd.DataFrame, name: str, game_id: str) -> BehaviorProfile:
    """Construit un BehaviorProfile depuis un groupe de sessions pandas."""
    n = len(group)

    def m(col):
        return float(group[col].mean()) if col in group.columns else 0.0

    def s(col):
        v = float(group[col].std()) if col in group.columns else 0.0
        return v if not np.isnan(v) else 0.0

    return BehaviorProfile(
        game_id=game_id,
        profile_name=name,
        n_sessions=n,
        lx_mean=m("lx_mean"),
        lx_std=m("lx_std"),
        ly_mean=m("ly_mean"),
        ly_std=m("ly_std"),
        rx_std=m("rx_std"),
        ry_std=m("ry_std"),
        lt_mean=m("lt_mean"),
        lt_std=s("lt_mean"),
        rt_mean=m("rt_mean"),
        rt_std=s("rt_mean"),
        btn_press_rate=m("btn_press_rate"),
        btn_hold_avg_ms=max(50.0, m("btn_hold_avg_ms")),
        btn_hold_std_ms=max(30.0, s("btn_hold_avg_ms")),
        btn_variety=m("btn_variety"),
        expected_score=m("score"),
        expected_duration=m("duration_sec"),
        input_regularity=m("input_regularity"),
    )


def load_profiles_from_csv(game_id: str, data_path: str = None) -> dict[str, BehaviorProfile]:
    """
    Charge les profils comportementaux depuis le CSV synthétique.
    Groupe par colonne 'profile' si elle existe, sinon par 'player_name'.
    """
    if data_path is None:
        data_path = _find_data_file()

    if data_path is None or not os.path.exists(data_path):
        print(f"⚠️  Données non trouvées — profils mock utilisés")
        return _mock_profiles(game_id)

    df      = pd.read_csv(data_path)
    df_game = df[df["game_id"] == game_id]

    if len(df_game) == 0:
        print(f"⚠️  Aucune session pour {game_id} — profils mock utilisés")
        return _mock_profiles(game_id)

    group_col = "profile" if "profile" in df_game.columns else "player_name"
    profiles  = {}
    for name, group in df_game.groupby(group_col):
        profiles[name] = _build_profile_from_group(group, name, game_id)
        print(f"✅ Profil CSV  : {name} ({len(group)} sessions) — score ≈ {profiles[name].expected_score:.0f}")

    return profiles


def load_profiles_from_supabase(game_id: str) -> dict[str, BehaviorProfile]:
    """
    Charge les profils depuis Supabase, groupés par player_name.
    Retourne un dict {player_name: BehaviorProfile}.
    Fallback sur mock si Supabase indisponible ou pas assez de données.
    """
    try:
        from core.supabase_client import fetch_all_sessions
        data = fetch_all_sessions()
        if not data:
            print("⚠️  Supabase vide — mock utilisé")
            return _mock_profiles(game_id)

        df      = pd.DataFrame(data)
        df_game = df[df["game_id"] == game_id]

        if len(df_game) == 0:
            print(f"⚠️  Aucune session Supabase pour {game_id} — mock utilisé")
            return _mock_profiles(game_id)

        profiles = {}
        for player_name, group in df_game.groupby("player_name"):
            n = len(group)
            if n < 2:
                print(f"⚠️  {player_name} : {n} session(s) pour {game_id} — ignoré")
                continue
            profiles[player_name] = _build_profile_from_group(group, player_name, game_id)
            print(f"✅ Profil réel : {player_name} ({n} sessions) — score ≈ {profiles[player_name].expected_score:.0f}")

        if not profiles:
            print("⚠️  Aucun joueur avec assez de sessions — mock utilisé")
            return _mock_profiles(game_id)

        return profiles

    except Exception as e:
        print(f"⚠️  Supabase erreur ({e}) — mock utilisé")
        return _mock_profiles(game_id)


def _mock_profiles(game_id: str) -> dict[str, BehaviorProfile]:
    """Profils de secours si aucune donnée disponible."""
    defaults = {
        "labyrinth": {
            "Explorateur": BehaviorProfile("labyrinth","Explorateur",0, lx_std=0.47,ly_std=0.44, btn_press_rate=0.001, expected_score=640),
            "Speedrunner": BehaviorProfile("labyrinth","Speedrunner",0, lx_std=0.65,ly_std=0.66, btn_press_rate=0.005, expected_score=995),
        },
        "shooter": {
            "Sniper":  BehaviorProfile("shooter","Sniper",0,  lx_std=0.42,ly_std=0.49, btn_press_rate=0.18, btn_hold_avg_ms=14000, expected_score=6400),
            "Rusheur": BehaviorProfile("shooter","Rusheur",0, lx_std=0.30,ly_std=0.47, btn_press_rate=0.24, btn_hold_avg_ms=1600,  expected_score=2800),
            "Fantôme": BehaviorProfile("shooter","Fantôme",0, lx_std=0.05,ly_std=0.05, btn_press_rate=0.02, btn_hold_avg_ms=50,    expected_score=200),
        },
        "reflex": {
            "Réactif":  BehaviorProfile("reflex","Réactif",0,  btn_press_rate=0.32, btn_hold_avg_ms=142, expected_score=700),
            "Hésitant": BehaviorProfile("reflex","Hésitant",0, btn_press_rate=0.19, btn_hold_avg_ms=195, expected_score=280),
        },
        "racing": {
            "Pilote maîtrisé": BehaviorProfile("racing","Pilote maîtrisé",0, lx_std=0.51,ly_std=0.65, btn_press_rate=0.001, expected_score=3400),
            "Pilote instable": BehaviorProfile("racing","Pilote instable",0, lx_std=0.53,ly_std=0.57, btn_press_rate=0.035, expected_score=1200),
        },
    }
    return defaults.get(game_id, {})


# ─────────────────────────────────────────────────────────────────────────────
# GÉNÉRATEUR D'INPUTS FRAME PAR FRAME
# ─────────────────────────────────────────────────────────────────────────────

class InputGenerator:
    """
    Génère des ControllerState synthétiques frame par frame.

    Logique par jeu :
    - labyrinth/racing : mouvement directionnel par segments (N/S/E/O)
    - shooter          : déplacement + trigger_right pour tirer + joystick droit pour viser
    - reflex           : boutons purs, joystick quasi nul
    """

    FPS = 30

    # Seuils exacts lus dans les jeux
    DEADZONE_JOYSTICK = 0.06   # labyrinth/shooter : abs(dx) > 0.05
    SHOOT_THRESHOLD   = 0.35   # shooter : trigger_right > 0.3

    def __init__(self, profile: BehaviorProfile, seed: int = None):
        self.profile  = profile
        self.game_id  = profile.game_id
        self.rng      = np.random.RandomState(seed)

        # ── Joystick gauche ──
        self.lx = 0.0
        self.ly = 0.0

        # Amplitude de déplacement selon lx_std du profil
        # On force une amplitude minimale > deadzone pour que le jeu réagisse
        self._move_amp = max(0.25, min(0.95, profile.lx_std * 1.4))

        # Durée d'un segment directionnel en frames (avant de changer de direction)
        # Profil actif = segments courts, profil prudent = segments longs
        seg_base = 45 if profile.lx_std > 0.55 else 70
        self._seg_frames     = seg_base
        self._seg_remaining  = self._seg_frames
        self._cur_dir        = self._new_direction()

        # ── Joystick droit (shooter : visée) ──
        self.rx = 0.0
        self.ry = 0.0
        self._aim_seg        = 20
        self._aim_remaining  = self._aim_seg
        self._aim_dir        = self._new_direction()

        # ── Gâchette droite (shooter : tir) ──
        # On utilise trigger_right > 0.3 comme dans shooter_game.py
        self._rt             = 0.0
        self._shoot_hold     = 0      # frames restantes de tir
        # Fréquence de tir selon btn_press_rate du profil
        self._shoot_prob     = max(0.04, min(0.25, profile.btn_press_rate * 0.8))
        hold_ms              = max(100, profile.btn_hold_avg_ms)
        self._shoot_hold_mean = max(3, int(hold_ms / (1000 / self.FPS)))

        # ── Boutons (reflex principalement) ──
        self._btn_hold_frames = {0: 0, 1: 0, 2: 0, 3: 0}
        self._btn_state       = {0: False, 1: False, 2: False, 3: False}
        n_btns                = max(1, round(profile.btn_variety * 4))
        self._active_buttons  = list(range(min(n_btns, 4)))
        self._btn_press_prob  = max(0.02, profile.btn_press_rate / max(1, len(self._active_buttons)))
        hold_frames_mean      = max(1, int(max(80, profile.btn_hold_avg_ms) / (1000 / self.FPS)))
        self._hold_frames_mean = hold_frames_mean
        self._hold_frames_std  = max(1, int(hold_frames_mean * 0.4))

        print(f"🤖 Agent [{self.game_id}] profil={profile.profile_name} | "
              f"move_amp={self._move_amp:.2f} | "
              f"shoot_prob={self._shoot_prob:.2f} | "
              f"btn_prob={self._btn_press_prob:.2f}")

    def _new_direction(self):
        """Tire une direction cardinale (N/S/E/O) ou diagonale."""
        dirs = [
            ( 1.0,  0.0), (-1.0,  0.0),
            ( 0.0,  1.0), ( 0.0, -1.0),
            ( 0.7,  0.7), (-0.7,  0.7),
            ( 0.7, -0.7), (-0.7, -0.7),
        ]
        return dirs[self.rng.randint(len(dirs))]

    def next_state(self) -> ControllerState:
        """Génère le prochain ControllerState selon le jeu."""

        # ════════════════════════════════════════════════
        # JOYSTICK GAUCHE — mouvement par segments
        # ════════════════════════════════════════════════
        self._seg_remaining -= 1
        if self._seg_remaining <= 0:
            self._cur_dir       = self._new_direction()
            self._seg_frames    = max(15, int(self.rng.normal(
                45 if self._move_amp > 0.6 else 70, 10
            )))
            self._seg_remaining = self._seg_frames

        # Amplitude avec léger bruit
        amp  = self._move_amp + self.rng.normal(0, 0.05)
        amp  = float(np.clip(amp, 0.15, 1.0))
        self.lx = float(np.clip(self._cur_dir[0] * amp, -1.0, 1.0))
        self.ly = float(np.clip(self._cur_dir[1] * amp, -1.0, 1.0))

        # ════════════════════════════════════════════════
        # JOYSTICK DROIT — visée (shooter uniquement)
        # ════════════════════════════════════════════════
        self._aim_remaining -= 1
        if self._aim_remaining <= 0:
            self._aim_dir       = self._new_direction()
            self._aim_remaining = max(10, int(self.rng.normal(20, 5)))

        if self.game_id == "shooter":
            aim_amp  = 0.6 + self.rng.normal(0, 0.1)
            self.rx  = float(np.clip(self._aim_dir[0] * aim_amp, -1.0, 1.0))
            self.ry  = float(np.clip(self._aim_dir[1] * aim_amp, -1.0, 1.0))
        else:
            self.rx = 0.0
            self.ry = 0.0

        # ════════════════════════════════════════════════
        # GÂCHETTE DROITE — tir (shooter : seuil > 0.3)
        # ════════════════════════════════════════════════
        if self.game_id == "shooter":
            if self._shoot_hold > 0:
                self._shoot_hold -= 1
                self._rt = 0.75  # bien au-dessus du seuil 0.3
            else:
                self._rt = 0.0
                if self.rng.random() < self._shoot_prob:
                    self._shoot_hold = max(2, int(self.rng.normal(
                        self._shoot_hold_mean, self._shoot_hold_mean * 0.3
                    )))
        else:
            self._rt = 0.0

        # ════════════════════════════════════════════════
        # BOUTONS — automate appui/relâchement
        # ════════════════════════════════════════════════
        for btn in self._active_buttons:
            if self._btn_hold_frames[btn] > 0:
                self._btn_hold_frames[btn] -= 1
                self._btn_state[btn] = True
            else:
                self._btn_state[btn] = False
                if self.rng.random() < self._btn_press_prob:
                    self._btn_hold_frames[btn] = max(1, int(self.rng.normal(
                        self._hold_frames_mean, self._hold_frames_std
                    )))
                    self._btn_state[btn] = True

        buttons = {i: self._btn_state[i] for i in range(4)}

        return ControllerState(
            timestamp=time.time(),
            axis_left_x=self.lx,
            axis_left_y=self.ly,
            axis_right_x=self.rx,
            axis_right_y=self.ry,
            trigger_left=0.0,
            trigger_right=self._rt,
            buttons=buttons,
            hat=(0, 0),
            source="agent",
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLASSE PRINCIPALE : GameAgent
# ─────────────────────────────────────────────────────────────────────────────



# ═══════════════════════════════════════════════════════════════════
# REPLAY AGENT — rejoue les séquences réelles frame par frame
# ═══════════════════════════════════════════════════════════════════

class SequenceReplayGenerator:
    """
    Rejoue les inputs enregistrés d'un joueur réel frame par frame,
    en ajoutant un bruit gaussien calibré pour éviter la répétition exacte.

    Stratégie :
      1. Charge toutes les frames de inputs_live pour ce joueur + jeu
      2. Sélectionne aléatoirement une session parmi celles disponibles
      3. Rejoue frame par frame avec bruit σ ≈ 0.04 sur les axes
      4. Quand une session est terminée → passe à une autre aléatoirement
    """

    # Bruit gaussien sur les axes (σ) — assez petit pour rester fidèle
    AXIS_NOISE   = 0.04
    TRIGGER_NOISE = 0.03

    def __init__(self, sequences: list[list[dict]], noise_level: float = 1.0):
        """
        sequences : liste de sessions, chaque session = liste de frames
                    [{"lx":..., "ly":..., "rt":..., "btn_a":..., ...}, ...]
        noise_level : multiplicateur du bruit (0 = replay exact, 1 = défaut, 2 = plus bruité)
        """
        if not sequences:
            raise ValueError("Aucune séquence disponible pour le replay")

        self.sequences   = sequences
        self.noise       = noise_level
        self.rng         = np.random.RandomState()

        # Sélectionner une session de départ aléatoire
        self._session_idx   = self.rng.randint(len(self.sequences))
        self._frame_idx     = 0
        self._current_seq   = self.sequences[self._session_idx]

        total_frames = sum(len(s) for s in sequences)
        print(f"🎬 SequenceReplayGenerator : {len(sequences)} session(s) · "
              f"{total_frames} frames · bruit σ={self.AXIS_NOISE * noise_level:.3f}")

    def _next_session(self):
        """Passe à une autre session aléatoirement."""
        self._session_idx = self.rng.randint(len(self.sequences))
        self._frame_idx   = 0
        self._current_seq = self.sequences[self._session_idx]

    def _add_noise(self, value: float, sigma: float) -> float:
        """Ajoute un bruit gaussien et clamp dans [-1, 1]."""
        return float(np.clip(value + self.rng.normal(0, sigma * self.noise), -1.0, 1.0))

    def _add_trigger_noise(self, value: float) -> float:
        """Ajoute du bruit sur une gâchette et clamp dans [0, 1]."""
        return float(np.clip(value + self.rng.normal(0, self.TRIGGER_NOISE * self.noise), 0.0, 1.0))

    @property
    def current_session_token(self) -> str:
        """Token de la session actuellement en cours de replay (issu des frames humaines)."""
        if self._current_seq:
            return self._current_seq[0].get("session_token", "")
        return ""

    def next_state(self) -> ControllerState:
        """Retourne le prochain ControllerState issu du replay."""
        # Si on a épuisé la session courante → en prendre une autre
        if self._frame_idx >= len(self._current_seq):
            self._next_session()

        frame = self._current_seq[self._frame_idx]
        self._frame_idx += 1

        # Axes avec bruit
        lx = self._add_noise(frame.get("lx", 0.0), self.AXIS_NOISE)
        ly = self._add_noise(frame.get("ly", 0.0), self.AXIS_NOISE)
        rx = self._add_noise(frame.get("rx", 0.0), self.AXIS_NOISE)
        ry = self._add_noise(frame.get("ry", 0.0), self.AXIS_NOISE)

        # Gâchettes avec bruit
        lt = self._add_trigger_noise(frame.get("lt", 0.0))
        rt = self._add_trigger_noise(frame.get("rt", 0.0))

        # Boutons : replay exact (les transitions on/off sont précises)
        buttons = {
            0: bool(frame.get("btn_a", False)),
            1: bool(frame.get("btn_b", False)),
            2: bool(frame.get("btn_x", False)),
            3: bool(frame.get("btn_y", False)),
        }

        return ControllerState(
            timestamp=time.time(),
            axis_left_x=lx,
            axis_left_y=ly,
            axis_right_x=rx,
            axis_right_y=ry,
            trigger_left=lt,
            trigger_right=rt,
            buttons=buttons,
            hat=(0, 0),
            source="agent_replay",
        )


def load_sequences_from_supabase(game_id: str, player_name: str) -> list[list[dict]]:
    """
    Charge les séquences de frames depuis inputs_live pour un joueur + jeu donnés.
    Retourne une liste de sessions, chaque session étant une liste de frames ordonnées.
    Retourne [] si Supabase indisponible ou aucune donnée.
    """
    try:
        from core.supabase_client import fetch_player_sequences
        sequences = fetch_player_sequences(game_id=game_id, player_name=player_name)
        if sequences:
            print(f"✅ Séquences chargées : {len(sequences)} session(s) pour {player_name} @ {game_id}")
            return sequences
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  fetch_player_sequences indisponible ({e})")

    # Fallback : essayer avec fetch_live_inputs filtré
    try:
        from core.supabase_client import fetch_live_inputs
        rows = fetch_live_inputs(limit=5000)
        if not rows:
            return []

        # Filtrer par joueur et jeu
        rows = [r for r in rows
                if r.get("player_name") == player_name
                and r.get("game_id") == game_id]

        if not rows:
            print(f"⚠️  Aucune frame pour {player_name} @ {game_id} dans inputs_live")
            return []

        # Grouper par session_token
        from collections import defaultdict
        sessions_dict = defaultdict(list)
        for row in rows:
            token = row.get("session_token", "unknown")
            sessions_dict[token].append(row)

        sequences = [frames for frames in sessions_dict.values() if len(frames) >= 10]
        print(f"✅ Séquences reconstituées : {len(sequences)} session(s) · "
              f"{sum(len(s) for s in sequences)} frames pour {player_name} @ {game_id}")
        return sequences

    except Exception as e:
        print(f"⚠️  Erreur chargement séquences ({e}) — fallback stats")
        return []

class GameAgent:
    """
    Agent imitateur — remplace le Controller dans BaseGame.
    Interface identique à Controller : possède une méthode get_state().

    Deux modes :
      mode="profile" : imite un profil générique depuis le CSV synthétique
      mode="player"  : imite un joueur réel depuis Supabase
    """

    def __init__(self, game_id: str, profile_name: str,
                 data_path: str = None, mode: str = "profile",
                 noise_level: float = 1.0):
        self.game_id      = game_id
        self.profile_name = profile_name
        self.mode         = mode

        print(f"\n🤖 Chargement agent : {profile_name} @ {game_id} [mode={mode}]")

        # ════════════════════════════════════════════
        # MODE PLAYER : Replay séquence réelle + bruit
        # ════════════════════════════════════════════
        if mode == "player":
            sequences = load_sequences_from_supabase(game_id, player_name=profile_name)

            if sequences:
                # On a des séquences réelles → replay fidèle
                self.generator = SequenceReplayGenerator(sequences, noise_level=noise_level)
                # Profil stats pour affichage (fallback)
                profiles = load_profiles_from_supabase(game_id)
                if profile_name in profiles:
                    self.profile = profiles[profile_name]
                else:
                    self.profile = _mock_profiles(game_id).get(
                        list(_mock_profiles(game_id).keys())[0]
                    )
                print(f"✅ Agent REPLAY prêt : {profile_name} | "
                      f"{len(sequences)} session(s) mémorisée(s) | "
                      f"bruit σ={SequenceReplayGenerator.AXIS_NOISE * noise_level:.3f}")
                return

            else:
                # Pas de séquences → fallback sur stats Supabase
                print(f"⚠️  Pas de séquences pour {profile_name} — fallback stats Supabase")
                profiles = load_profiles_from_supabase(game_id)
                if profile_name not in profiles:
                    print(f"⚠️  {profile_name} absent Supabase — fallback CSV")
                    profiles = load_profiles_from_csv(game_id, data_path)

        # ════════════════════════════════════════════
        # MODE PROFILE : Générateur statistique
        # ════════════════════════════════════════════
        else:
            profiles = load_profiles_from_csv(game_id, data_path)

        # Fallback : premier profil disponible
        if profile_name not in profiles:
            available = list(profiles.keys())
            print(f"⚠️  '{profile_name}' inconnu. Disponibles : {available}")
            profile_name      = available[0] if available else profile_name
            self.profile_name = profile_name

        self.profile   = profiles[profile_name]
        self.generator = InputGenerator(self.profile)

        src = "stats Supabase" if mode == "player" else "CSV synthétique"
        print(f"✅ Agent STATS prêt [{src}] : {self.profile_name} | "
              f"score attendu ≈ {self.profile.expected_score:.0f} | "
              f"sessions apprises : {self.profile.n_sessions}")

    def get_state(self) -> ControllerState:
        """Interface identique à Controller.get_state()"""
        return self.generator.next_state()

    def is_connected(self) -> bool:
        return True

    def reconnect(self):
        pass

    @staticmethod
    def list_profiles(game_id: str, data_path: str = None) -> list[str]:
        """Liste les profils CSV disponibles pour un jeu."""
        return list(load_profiles_from_csv(game_id, data_path).keys())

    @staticmethod
    def list_real_players(game_id: str) -> list[str]:
        """Liste les joueurs réels disponibles dans Supabase pour un jeu."""
        profiles = load_profiles_from_supabase(game_id)
        return list(profiles.keys())


