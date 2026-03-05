"""
Session Recorder — enregistre une session de jeu et extrait les features
"""

import csv
import time
import os
import math
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Optional
from core.controller import ControllerState


@dataclass
class SessionFeatures:
    """Features extraites d'une session complète — utilisées pour le clustering"""

    player_name: str
    game_id: str
    duration_sec: float

    # --- Boutons ---
    btn_press_rate: float  # Appuis par seconde (tous boutons)
    btn_variety: float  # Nb de boutons différents utilisés / total
    btn_hold_avg_ms: float  # Durée moyenne d'appui en ms

    # --- Joystick gauche ---
    lx_mean: float  # Position moyenne X
    ly_mean: float  # Position moyenne Y
    lx_std: float  # Variabilité X (agitation)
    ly_std: float  # Variabilité Y
    lx_direction_changes: float  # Nb de changements de direction / sec

    # --- Joystick droit ---
    rx_mean: float
    ry_mean: float
    rx_std: float
    ry_std: float

    # --- Gâchettes ---
    lt_mean: float  # Utilisation moyenne gâchette gauche
    rt_mean: float
    lt_brutality: float  # Δ moyen entre frames (douceur vs brutalité)
    rt_brutality: float

    # --- Timing & rythme ---
    reaction_time_avg_ms: float  # Temps moyen entre stimuli et réponse (si applicable)
    input_regularity: (
        float  # Écart-type des intervalles entre inputs (0 = très régulier)
    )

    # --- Source ---
    source: str = "unknown"  # "controller" ou "keyboard"

    # --- Score ---
    score: int = 0


class SessionRecorder:
    """
    Enregistre les états manette pendant une session
    et calcule les features à la fin.
    """

    SAMPLE_RATE_HZ = 30  # Fréquence d'échantillonnage

    def __init__(self, player_name: str, game_id: str):
        self.player_name = player_name
        self.game_id = game_id
        self.states: List[ControllerState] = []
        self.events: List[dict] = []  # Événements discrets (appui bouton, etc.)
        self._btn_press_times: dict = {}  # Pour mesurer la durée d'appui
        self._btn_hold_durations: List[float] = []
        self._last_btn_state: dict = {}
        self._input_timestamps: List[float] = []  # Pour mesurer la régularité
        self.score = 0
        self.start_time: Optional[float] = None

    def start(self):
        self.start_time = time.time()
        print(f"🔴 Session démarrée pour {self.player_name} ({self.game_id})")

    def record(self, state: ControllerState):
        """À appeler à chaque frame avec l'état manette courant"""
        self.states.append(state)
        self._detect_button_events(state)

    def _detect_button_events(self, state: ControllerState):
        """Détecte les appuis/relâchements et mesure les durées"""
        for btn_id, pressed in state.buttons.items():
            was_pressed = self._last_btn_state.get(btn_id, False)

            if pressed and not was_pressed:
                # Nouvel appui
                self._btn_press_times[btn_id] = state.timestamp
                self._input_timestamps.append(state.timestamp)
                self.events.append(
                    {
                        "type": "btn_press",
                        "button": btn_id,
                        "timestamp": state.timestamp,
                    }
                )

            elif not pressed and was_pressed:
                # Relâchement
                if btn_id in self._btn_press_times:
                    duration_ms = (
                        state.timestamp - self._btn_press_times[btn_id]
                    ) * 1000
                    self._btn_hold_durations.append(duration_ms)
                    del self._btn_press_times[btn_id]

        self._last_btn_state = dict(state.buttons)

    def add_score(self, points: int):
        self.score += points

    def stop(self) -> SessionFeatures:
        """Arrête la session et calcule les features"""
        if not self.states:
            raise ValueError("Aucun état enregistré !")

        duration = self.states[-1].timestamp - self.states[0].timestamp
        if duration == 0:
            duration = 0.001

        features = self._compute_features(duration)
        print(f"✅ Session terminée — {len(self.states)} frames, {duration:.1f}s")
        return features

    def _std(self, values):
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))

    def _mean(self, values):
        return sum(values) / len(values) if values else 0.0

    def _compute_features(self, duration: float) -> SessionFeatures:
        states = self.states

        # Axes
        lx = [s.axis_left_x for s in states]
        ly = [s.axis_left_y for s in states]
        rx = [s.axis_right_x for s in states]
        ry = [s.axis_right_y for s in states]
        lt = [s.trigger_left for s in states]
        rt = [s.trigger_right for s in states]

        # Changements de direction joystick gauche
        lx_dir_changes = (
            sum(
                1
                for i in range(1, len(lx))
                if lx[i] * lx[i - 1] < 0  # Changement de signe
            )
            / duration
        )

        # Brutalité gâchettes (variation absolue entre frames)
        lt_brutality = self._mean([abs(lt[i] - lt[i - 1]) for i in range(1, len(lt))])
        rt_brutality = self._mean([abs(rt[i] - rt[i - 1]) for i in range(1, len(rt))])

        # Boutons
        press_events = [e for e in self.events if e["type"] == "btn_press"]
        buttons_used = set(e["button"] for e in press_events)
        total_buttons = max(len(states[0].buttons), 1) if states else 1

        # Régularité des inputs
        if len(self._input_timestamps) > 1:
            intervals = [
                self._input_timestamps[i] - self._input_timestamps[i - 1]
                for i in range(1, len(self._input_timestamps))
            ]
            input_regularity = self._std(intervals)
        else:
            input_regularity = 0.0

        # Source : controller si au moins 80% des frames viennent d'une manette
        kb_count = sum(1 for s in states if s.source == "keyboard")
        source = "keyboard" if kb_count / len(states) > 0.5 else "controller"

        return SessionFeatures(
            player_name=self.player_name,
            game_id=self.game_id,
            duration_sec=duration,
            btn_press_rate=len(press_events) / duration,
            btn_variety=len(buttons_used) / total_buttons,
            btn_hold_avg_ms=self._mean(self._btn_hold_durations),
            lx_mean=self._mean(lx),
            ly_mean=self._mean(ly),
            lx_std=self._std(lx),
            ly_std=self._std(ly),
            lx_direction_changes=lx_dir_changes,
            rx_mean=self._mean(rx),
            ry_mean=self._mean(ry),
            rx_std=self._std(rx),
            ry_std=self._std(ry),
            lt_mean=self._mean(lt),
            rt_mean=self._mean(rt),
            lt_brutality=lt_brutality,
            rt_brutality=rt_brutality,
            reaction_time_avg_ms=getattr(self, 'reaction_times_avg', 0.0),
            input_regularity=input_regularity,
            source=source,
            score=self.score,
        )


def save_features_to_csv(
    features: SessionFeatures, filepath: str = "data/sessions.csv"
):
    """Sauvegarde les features dans un CSV cumulatif"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    row = asdict(features)
    file_exists = os.path.exists(filepath)

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"💾 Features sauvegardées dans {filepath}")
