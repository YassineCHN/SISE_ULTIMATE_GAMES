"""
BaseGame — classe abstraite dont héritent tous les mini-jeux
Chaque jeu doit implémenter : setup(), update(), draw(), is_over()

Supporte deux modes :
  - Mode humain  : controller = Controller()  (manette ou clavier)
  - Mode agent   : controller = GameAgent(game_id, profile_name)
"""

from time import time

from core.supabase_client import save_features_to_supabase
import pygame
from abc import ABC, abstractmethod
from core.controller import Controller, ControllerState
from core.recorder import SessionRecorder, save_features_to_csv
import time
import threading
from core.supabase_client import save_features_to_supabase, send_inputs_batch


class BaseGame(ABC):
    """
    Classe de base pour tous les mini-jeux.
    Gère la boucle principale, la manette/agent et l'enregistrement.

    Paramètres
    ----------
    player_name : str
        Nom du joueur (ou "Agent_IA" si mode agent).
    headless : bool
        Mode sans fenêtre pour les tests automatisés.
    agent : GameAgent | None
        Si fourni, l'agent remplace la manette humaine.
        L'agent expose la même interface que Controller (get_state()).
    """

    FPS = 30
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    BG_COLOR = (15, 15, 30)

    # Couleur HUD différente pour l'agent (violet) vs humain (blanc)
    HUD_COLOR_HUMAN = (200, 200, 200)
    HUD_COLOR_AGENT = (180, 100, 255)

    def __init__(self, player_name: str, headless: bool = False, agent=None):
        self.player_name = player_name
        self.headless = headless
        self.agent = agent  # None → mode humain

        # Controller : agent si fourni, sinon vrai controller
        if self.agent is not None:
            self.controller = self.agent
            print(f"🤖 Mode agent activé : {self.agent.profile_name}")
        else:
            self.controller = Controller()

        self.recorder = SessionRecorder(player_name, game_id=self.game_id)
        self.running = False
        self.clock = pygame.time.Clock()
        self.screen = None
        self.font = None

    # ── Interface abstraite ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def game_id(self) -> str:
        """Identifiant unique du jeu (ex: 'reflex', 'labyrinth')"""
        pass

    @abstractmethod
    def setup(self):
        """Initialisation des éléments du jeu (appelé une fois au démarrage)"""
        pass

    @abstractmethod
    def update(self, state: ControllerState, dt: float):
        """
        Logique de jeu à chaque frame.
        dt = delta time en secondes depuis la dernière frame.
        """
        pass

    @abstractmethod
    def draw(self, screen: pygame.Surface):
        """Rendu graphique de la frame courante"""
        pass

    @abstractmethod
    def is_over(self) -> bool:
        """Retourne True quand la partie est terminée"""
        pass

    def on_game_over(self):
        """Hook appelé à la fin de la partie (override si besoin)"""
        pass

    # ── Boucle principale ─────────────────────────────────────────────────────

    def run(self) -> dict:
        """
        Boucle principale du jeu.
        Retourne les features de la session sous forme de dict.
        """
        if not self.headless:
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.SCREEN_WIDTH, self.SCREEN_HEIGHT)
            )
            mode_label = (
                f"AGENT — {self.agent.profile_name}" if self.agent else self.game_id
            )
            pygame.display.set_caption(f"SISE ULTIMATE — {mode_label}")
            self.font = pygame.font.SysFont("monospace", 20)

        self._session_token = f"{self.player_name}_{int(time.time())}"
        self.setup()
        self.recorder.start()
        self._inputs_buffer = []
        self._last_flush = time.time()
        self.running = True

        while self.running:
            dt = self.clock.tick(self.FPS) / 1000.0

            # Gestion événements pygame (fermeture fenêtre)
            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    # Touche ECHAP pour stopper l'agent
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            print("⏹  Agent stoppé par l'utilisateur (Echap)")
                            self.running = False

            # ── Lecture inputs : agent ou humain ──────────────────────────
            controller_state = self.controller.get_state()

            # Enregistrement
            self.recorder.record(controller_state)

            # Bufferiser pour Supabase inputs_live
            self._inputs_buffer.append(
                {
                    "player_name": self.player_name,
                    "game_id": self.game_id,
                    "session_token": self._session_token,
                    "lx": controller_state.axis_left_x,
                    "ly": controller_state.axis_left_y,
                    "rx": controller_state.axis_right_x,
                    "ry": controller_state.axis_right_y,
                    "lt": controller_state.trigger_left,
                    "rt": controller_state.trigger_right,
                    "btn_a": bool(controller_state.buttons.get(0, False)),
                    "btn_b": bool(controller_state.buttons.get(1, False)),
                    "btn_x": bool(controller_state.buttons.get(2, False)),
                    "btn_y": bool(controller_state.buttons.get(3, False)),
                    "event_type": controller_state.source,  # "agent" | "keyboard" | "controller"
                }
            )

            # Flush toutes les 500ms
            if time.time() - self._last_flush >= 0.5:
                if self._inputs_buffer:
                    threading.Thread(
                        target=send_inputs_batch,
                        args=(self._inputs_buffer.copy(),),
                        daemon=True,
                    ).start()
                    self._inputs_buffer.clear()
                self._last_flush = time.time()

            # Logique jeu
            self.update(controller_state, dt)

            # Rendu
            if not self.headless and self.screen:
                self.screen.fill(self.BG_COLOR)
                self.draw(self.screen)
                self._draw_hud(self.screen)
                # Bandeau agent visible à l'écran
                if self.agent:
                    self._draw_agent_banner(self.screen)
                pygame.display.flip()

            # Fin de partie ?
            if self.is_over():
                self.running = False

        self.on_game_over()
        features = self.recorder.stop()
        save_features_to_csv(features)
        save_features_to_supabase(features)

        if not self.headless:
            self._show_game_over_screen()
            pygame.quit()

        return features

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, screen: pygame.Surface):
        """HUD commun : score, joueur, temps"""
        if self.font is None:
            return
        elapsed = (
            self.recorder.states[-1].timestamp - self.recorder.states[0].timestamp
            if self.recorder.states
            else 0
        )
        hud_color = self.HUD_COLOR_AGENT if self.agent else self.HUD_COLOR_HUMAN
        texts = [
            f"Joueur : {self.player_name}",
            f"Score  : {self.recorder.score}",
            f"Temps  : {elapsed:.1f}s",
        ]
        for i, text in enumerate(texts):
            surf = self.font.render(text, True, hud_color)
            screen.blit(surf, (10, 10 + i * 24))

    def _draw_agent_banner(self, screen: pygame.Surface):
        """Bandeau en haut de l'écran indiquant que l'agent joue."""
        if self.font is None:
            return
        banner_text = f"🤖 AGENT IA — {self.agent.profile_name}  |  Echap pour arrêter"
        surf = self.font.render(banner_text, True, (180, 100, 255))
        # Fond semi-transparent
        banner_w = surf.get_width() + 20
        banner_h = surf.get_height() + 10
        banner_surf = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
        banner_surf.fill((30, 0, 60, 180))
        x = self.SCREEN_WIDTH // 2 - banner_w // 2
        screen.blit(banner_surf, (x, 0))
        screen.blit(surf, (x + 10, 5))

    def _show_game_over_screen(self):
        """Écran de fin de partie"""
        if self.screen is None:
            return
        self.screen.fill((10, 10, 20))
        font_big = pygame.font.SysFont("monospace", 48, bold=True)
        font_small = pygame.font.SysFont("monospace", 24)

        # Titre différent selon mode
        title = "AGENT — FIN DE PARTIE" if self.agent else "GAME OVER"
        title_col = (180, 100, 255) if self.agent else (255, 80, 80)

        surf = font_big.render(title, True, title_col)
        screen_cx = self.SCREEN_WIDTH // 2
        self.screen.blit(surf, (screen_cx - surf.get_width() // 2, 180))

        surf2 = font_small.render(
            f"Score final : {self.recorder.score}", True, (255, 255, 100)
        )
        self.screen.blit(surf2, (screen_cx - surf2.get_width() // 2, 280))

        if self.agent:
            surf3 = font_small.render(
                f"Profil imité : {self.agent.profile_name}", True, (180, 100, 255)
            )
            self.screen.blit(surf3, (screen_cx - surf3.get_width() // 2, 330))

        surf4 = font_small.render("Fermeture automatique...", True, (150, 150, 150))
        self.screen.blit(surf4, (screen_cx - surf4.get_width() // 2, 400))

        pygame.display.flip()
        pygame.time.wait(3000)
