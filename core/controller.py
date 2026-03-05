"""
Controller module — lecture temps réel de la manette via pygame
Compatible Xbox, PS4/PS5, manettes génériques USB
Fallback clavier complet quand aucune manette n'est branchée.

Clavier (mode sans manette) :
  Joystick gauche  : Flèches directionnelles
  Boutons A/B/X/Y  : Z / X / C / V
  Gâchette gauche  : A
  Gâchette droite  : E
"""

import pygame
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ControllerState:
    """Snapshot de l'état de la manette à un instant T"""

    timestamp: float
    axis_left_x: float = 0.0
    axis_left_y: float = 0.0
    axis_right_x: float = 0.0
    axis_right_y: float = 0.0
    trigger_left: float = 0.0
    trigger_right: float = 0.0
    buttons: dict = field(default_factory=dict)
    hat: tuple = (0, 0)
    source: str = "controller"  # "controller" ou "keyboard"
    button_l1: bool = False  # L1 / LB — index varie selon la manette
    button_r1: bool = False  # R1 / RB — index varie selon la manette


class Controller:
    AXIS_MAP = {
        "xbox":    {"lx": 0, "ly": 1, "rx": 3, "ry": 4, "lt": 2, "rt": 5},
        "ps3":     {"lx": 0, "ly": 1, "rx": 4, "ry": 5, "lt": 2, "rt": 3},  # DualShock 3 DirectInput
        "ps":      {"lx": 0, "ly": 1, "rx": 2, "ry": 3, "lt": 4, "rt": 5},  # PS4/PS5
        "generic": {"lx": 0, "ly": 1, "rx": 2, "ry": 3, "lt": 4, "rt": 5},
    }
    # Indices des boutons L1/R1 selon le type de manette
    L1R1_MAP = {
        "xbox":    (4, 5),   # LB, RB
        "ps3":     (10, 11), # L1, R1 sur DualShock 3 DirectInput
        "ps":      (4, 5),   # L1, R1 sur PS4/PS5
        "generic": (4, 5),
    }
    DEADZONE = 0.08

    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.joystick = None
        self.controller_type = "generic"
        self._axis_map = self.AXIS_MAP["generic"]
        self._connect()

    def _connect(self) -> bool:
        count = pygame.joystick.get_count()
        if count == 0:
            print("⚠️  Aucune manette détectée — mode clavier activé.")
            print("   Flèches = joystick | Z/X/C/V = boutons | A/E = gâchettes")
            return False
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        name = self.joystick.get_name().lower()
        print(f"🎮 Manette détectée : {self.joystick.get_name()}")
        if "xbox" in name or "xinput" in name:
            self.controller_type = "xbox"
        elif any(k in name for k in ("ps3", "dualshock 3", "playstation(r)3", "sixaxis")):
            self.controller_type = "ps3"  # Détecté avant "ps" (plus spécifique)
        elif any(k in name for k in ("playstation", "dualshock", "dualsense", "ps4", "ps5", "sony", "wireless controller")):
            self.controller_type = "ps"
        self._axis_map = self.AXIS_MAP[self.controller_type]
        n_axes = self.joystick.get_numaxes()
        n_btns = self.joystick.get_numbuttons()
        print(f"   Type détecté : {self.controller_type}  |  axes={n_axes}  boutons={n_btns}")
        return True

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.DEADZONE else value

    def _get_keyboard_state(self) -> ControllerState:
        """Simule un ControllerState complet depuis le clavier"""
        keys = pygame.key.get_pressed()
        lx = (1.0 if keys[pygame.K_RIGHT] else 0.0) - (
            1.0 if keys[pygame.K_LEFT] else 0.0
        )
        ly = (1.0 if keys[pygame.K_DOWN] else 0.0) - (1.0 if keys[pygame.K_UP] else 0.0)
        mag = (lx**2 + ly**2) ** 0.5
        if mag > 1.0:
            lx /= mag
            ly /= mag
        buttons = {
            0: bool(keys[pygame.K_z]),
            1: bool(keys[pygame.K_x]),
            2: bool(keys[pygame.K_c]),
            3: bool(keys[pygame.K_v]),
            4: bool(keys[pygame.K_SPACE]),
            5: bool(keys[pygame.K_LSHIFT]),
        }
        hat_x = (1 if keys[pygame.K_RIGHT] else 0) - (1 if keys[pygame.K_LEFT] else 0)
        hat_y = (1 if keys[pygame.K_UP] else 0) - (1 if keys[pygame.K_DOWN] else 0)
        return ControllerState(
            timestamp=time.time(),
            axis_left_x=lx,
            axis_left_y=ly,
            trigger_left=1.0 if keys[pygame.K_a] else 0.0,
            trigger_right=1.0 if keys[pygame.K_e] else 0.0,
            buttons=buttons,
            hat=(hat_x, hat_y),
            source="keyboard",
            button_l1=bool(keys[pygame.K_LSHIFT]),
            button_r1=bool(keys[pygame.K_SPACE]),
        )

    def get_state(self) -> ControllerState:
        pygame.event.pump()
        if self.joystick is None:
            return self._get_keyboard_state()
        m = self._axis_map
        num_axes = self.joystick.get_numaxes()

        def safe_axis(idx):
            return (
                self._apply_deadzone(self.joystick.get_axis(idx))
                if idx < num_axes
                else 0.0
            )

        # ── Lecture des gâchettes ──────────────────────────────────────
        lt_raw = safe_axis(m["lt"])
        rt_raw = safe_axis(m["rt"])  # 0.0 si axe hors plage

        if self.controller_type == "xbox" and num_axes <= 5:
            # Xbox 360 DirectInput : gâchettes combinées sur l'axe 2
            # LT = côté négatif, RT = côté positif
            combined = safe_axis(2)
            lt = max(0.0, -combined)
            rt = max(0.0, combined)
        elif m["rt"] >= num_axes:
            # Axe RT hors plage : gâchettes probablement combinées sur l'axe LT
            # L2 = côté positif, R2 = côté négatif
            lt = max(0.0, lt_raw)
            rt = max(0.0, -lt_raw)
        else:
            # Normalisation universelle : gère [-1,1] et [0,1]
            lt = (lt_raw + 1) / 2 if lt_raw < -0.5 else max(0.0, lt_raw)
            rt = (rt_raw + 1) / 2 if rt_raw < -0.5 else max(0.0, rt_raw)

        buttons = {
            i: bool(self.joystick.get_button(i))
            for i in range(self.joystick.get_numbuttons())
        }
        hat = self.joystick.get_hat(0) if self.joystick.get_numhats() > 0 else (0, 0)
        l1_idx, r1_idx = self.L1R1_MAP[self.controller_type]
        return ControllerState(
            timestamp=time.time(),
            axis_left_x=safe_axis(m["lx"]),
            axis_left_y=safe_axis(m["ly"]),
            axis_right_x=safe_axis(m["rx"]),
            axis_right_y=safe_axis(m["ry"]),
            trigger_left=lt,
            trigger_right=rt,
            buttons=buttons,
            hat=hat,
            source="controller",
            button_l1=buttons.get(l1_idx, False),
            button_r1=buttons.get(r1_idx, False),
        )

    def is_connected(self) -> bool:
        return self.joystick is not None

    def reconnect(self):
        pygame.joystick.quit()
        pygame.joystick.init()
        self._connect()
