"""
BaseGame — classe abstraite dont héritent tous les mini-jeux
Chaque jeu doit implémenter : setup(), update(), draw(), is_over()
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
    Gère la boucle principale, la manette et l'enregistrement.
    """

    FPS = 30
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 600
    BG_COLOR = (15, 15, 30)

    def __init__(self, player_name: str, headless: bool = False):
        self.player_name = player_name
        self.headless = headless  # Mode sans fenêtre pour les tests
        self.controller = Controller()
        self.recorder = SessionRecorder(player_name, game_id=self.game_id)
        self.running = False
        self.clock = pygame.time.Clock()
        self.screen = None
        self.font = None

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
            pygame.display.set_caption(f"SISE ULTIMATE — {self.game_id}")
            self.font = pygame.font.SysFont("monospace", 20)

        self.setup()
        self.recorder.start()
        self._inputs_buffer = []
        self._last_flush = time.time()
        self._session_token = f"{self.player_name}_{int(time.time())}"
        self.running = True

        while self.running:
            dt = self.clock.tick(self.FPS) / 1000.0  # Delta time en secondes

            # Gestion événements pygame (fermeture fenêtre)
            if not self.headless:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False

            # Lecture manette
            controller_state = self.controller.get_state()

            # Enregistrement
            self.recorder.record(controller_state)

            # Bufferiser l'input courant
            self._inputs_buffer.append({
                "player_name":   self.player_name,
                "game_id":       self.game_id,
                "session_token": self._session_token,
                "lx":  controller_state.axis_left_x,
                "ly":  controller_state.axis_left_y,
                "rx":  controller_state.axis_right_x,
                "ry":  controller_state.axis_right_y,
                "lt":  controller_state.trigger_left,
                "rt":  controller_state.trigger_right,
                "btn_a": bool(controller_state.buttons.get(0, False)),
                "btn_b": bool(controller_state.buttons.get(1, False)),
                "btn_x": bool(controller_state.buttons.get(2, False)),
                "btn_y": bool(controller_state.buttons.get(3, False)),
                "event_type": controller_state.source,
            })

            # Flush toutes les 500ms
            if time.time() - self._last_flush >= 0.5:
                if self._inputs_buffer:
                    threading.Thread(
                        target=send_inputs_batch,
                        args=(self._inputs_buffer.copy(),),
                        daemon=True
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

    def _draw_hud(self, screen: pygame.Surface):
        """HUD commun : score, joueur, temps"""
        if self.font is None:
            return
        elapsed = (
            self.recorder.states[-1].timestamp - self.recorder.states[0].timestamp
            if self.recorder.states
            else 0
        )
        texts = [
            f"Joueur : {self.player_name}",
            f"Score  : {self.recorder.score}",
            f"Temps  : {elapsed:.1f}s",
        ]
        for i, text in enumerate(texts):
            surf = self.font.render(text, True, (200, 200, 200))
            screen.blit(surf, (10, 10 + i * 24))

    def _show_game_over_screen(self):
        """Écran de fin de partie"""
        if self.screen is None:
            return
        self.screen.fill((10, 10, 20))
        font_big = pygame.font.SysFont("monospace", 48, bold=True)
        font_small = pygame.font.SysFont("monospace", 24)

        surf = font_big.render("GAME OVER", True, (255, 80, 80))
        self.screen.blit(surf, (self.SCREEN_WIDTH // 2 - surf.get_width() // 2, 200))

        surf2 = font_small.render(
            f"Score final : {self.recorder.score}", True, (255, 255, 100)
        )
        self.screen.blit(surf2, (self.SCREEN_WIDTH // 2 - surf2.get_width() // 2, 300))

        surf3 = font_small.render(
            "Appuyez sur une touche pour quitter", True, (150, 150, 150)
        )
        self.screen.blit(surf3, (self.SCREEN_WIDTH // 2 - surf3.get_width() // 2, 380))

        pygame.display.flip()
        pygame.time.wait(3000)
