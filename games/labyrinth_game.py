"""
LabyrinthGame — Navigation dans un labyrinthe généré procéduralement
Le joueur utilise le joystick gauche (ou flèches) pour se déplacer.
Features capturées : style de navigation, hésitations, vitesse, exploration.
"""

import pygame
import random
import time
from collections import deque
from games.base_game import BaseGame
from core.controller import ControllerState

# Couleurs
C_WALL = (30, 30, 50)
C_PATH = (60, 60, 90)
C_PLAYER = (80, 200, 120)
C_EXIT = (255, 200, 50)
C_VISITED = (40, 40, 70)
C_TRAIL = (50, 100, 80)
C_HUD_BG = (10, 10, 20)


class LabyrinthGame(BaseGame):
    """
    Labyrinthe généré par DFS. Le joueur doit atteindre la sortie.
    3 labyrinthes à compléter (de taille croissante) ou 90 secondes max.
    """

    GAME_DURATION = 90
    CELL_SIZE = 32
    COLS = 19  # doit être impair
    ROWS = 15  # doit être impair
    SPEED = 4  # pixels par frame au maximum

    @property
    def game_id(self) -> str:
        return "labyrinth"

    def _maze_seed(self, maze_number: int = 0) -> int:
        """Calcule un seed déterministe pour la génération du labyrinthe.

        En mode replay agent : utilise le session_token du joueur humain original
        → l'agent rejoue exactement le même labyrinthe que le joueur enregistré.
        En mode humain / stats : utilise le session_token de la session courante.
        """
        if self.agent is not None:
            gen = getattr(self.agent, "generator", None)
            if gen is not None and hasattr(gen, "current_session_token"):
                token = gen.current_session_token
                if token:
                    return hash(f"{token}_{maze_number}") % (2 ** 31)
        token = getattr(self, "_session_token", "")
        return hash(f"{token}_{maze_number}") % (2 ** 31)

    def setup(self):
        self.time_elapsed = 0.0
        self.mazes_completed = 0
        self.total_mazes = 3
        self._maze_number = 0
        self._generate_maze(seed=self._maze_seed(0))

        # Stats navigation
        self.total_distance = 0.0
        self.direction_changes = 0
        self.hesitation_frames = 0  # Frames sans mouvement alors que possible
        self.backtrack_count = 0
        self._last_direction = (0, 0)
        self._visited_cells = set()
        self._trail = deque(maxlen=30)  # Historique de position pour affichage
        self._move_buffer = (0.0, 0.0)  # Sous-pixel movement

    def _generate_maze(self, seed: int = None):
        """Génère un labyrinthe par DFS récursif. Si seed fourni, génération reproductible."""
        if seed is not None:
            random.seed(seed)
        cols, rows = self.COLS, self.ROWS
        # Grille : True = mur, False = couloir
        self.grid = [[True] * cols for _ in range(rows)]

        def carve(cx, cy):
            self.grid[cy][cx] = False
            dirs = [(0, -2), (0, 2), (-2, 0), (2, 0)]
            random.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < cols and 0 <= ny < rows and self.grid[ny][nx]:
                    self.grid[cy + dy // 2][cx + dx // 2] = False
                    carve(nx, ny)

        carve(1, 1)

        # Offset calculé EN PREMIER pour que la position joueur soit correcte
        maze_w = cols * self.CELL_SIZE
        maze_h = rows * self.CELL_SIZE
        self.offset_x = (self.SCREEN_WIDTH - maze_w) // 2
        self.offset_y = (self.SCREEN_HEIGHT - maze_h) // 2 + 10

        # Position joueur et sortie (coordonnées pixel avec offset)
        self.player_x = float(self.offset_x + 1 * self.CELL_SIZE + self.CELL_SIZE // 2)
        self.player_y = float(self.offset_y + 1 * self.CELL_SIZE + self.CELL_SIZE // 2)
        self.exit_cell = (cols - 2, rows - 2)
        self.grid[rows - 2][cols - 2] = False

        self._visited_cells = set()
        self._visited_cells.add((1, 1))
        self._trail = deque(maxlen=30)
        self._move_buffer = (0.0, 0.0)

    def _cell_of(self, px, py):
        """Convertit une position pixel en coordonnée cellule"""
        cx = int((px - self.offset_x) / self.CELL_SIZE)
        cy = int((py - self.offset_y) / self.CELL_SIZE)
        return cx, cy

    def _is_wall(self, px, py, margin=6):
        """Vérifie si un point (avec marge) touche un mur"""
        for corner_x in [px - margin, px + margin]:
            for corner_y in [py - margin, py + margin]:
                cx = int((corner_x - self.offset_x) / self.CELL_SIZE)
                cy = int((corner_y - self.offset_y) / self.CELL_SIZE)
                if cx < 0 or cy < 0 or cx >= self.COLS or cy >= self.ROWS:
                    return True
                if self.grid[cy][cx]:
                    return True
        return False

    def update(self, state: ControllerState, dt: float):
        self.time_elapsed += dt

        # --- Mouvement ---
        dx = state.axis_left_x
        dy = state.axis_left_y

        # Fallback D-pad si joystick neutre
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            hx, hy = state.hat
            dx = float(hx)
            dy = float(-hy)  # hat Y est inversé

        speed = self.SPEED * (1 + abs(dx) * 0.5 + abs(dy) * 0.5)
        moved = False

        new_x = self.player_x + dx * speed
        new_y = self.player_y + dy * speed

        # Mouvement axe X
        if abs(dx) > 0.05 and not self._is_wall(new_x, self.player_y):
            dist = abs(new_x - self.player_x)
            self.player_x = new_x
            self.total_distance += dist
            moved = True

        # Mouvement axe Y
        if abs(dy) > 0.05 and not self._is_wall(self.player_x, new_y):
            dist = abs(new_y - self.player_y)
            self.player_y = new_y
            self.total_distance += dist
            moved = True

        # --- Stats navigation ---
        direction = (
            1 if dx > 0.1 else (-1 if dx < -0.1 else 0),
            1 if dy > 0.1 else (-1 if dy < -0.1 else 0),
        )

        if moved:
            if direction != self._last_direction and self._last_direction != (0, 0):
                self.direction_changes += 1
            self._last_direction = direction
            self._trail.append((int(self.player_x), int(self.player_y)))

            # Cellule visitée
            cx, cy = self._cell_of(self.player_x, self.player_y)
            if (cx, cy) in self._visited_cells:
                self.backtrack_count += 1
            else:
                self._visited_cells.add((cx, cy))
                self.recorder.add_score(2)  # Points pour exploration
        else:
            self.hesitation_frames += 1

        # --- Arrivée à la sortie ---
        ex, ey = self.exit_cell
        exit_px = self.offset_x + ex * self.CELL_SIZE + self.CELL_SIZE // 2
        exit_py = self.offset_y + ey * self.CELL_SIZE + self.CELL_SIZE // 2
        dist_to_exit = (
            (self.player_x - exit_px) ** 2 + (self.player_y - exit_py) ** 2
        ) ** 0.5

        if dist_to_exit < self.CELL_SIZE * 0.6:
            self.mazes_completed += 1
            time_bonus = max(0, int((self.GAME_DURATION - self.time_elapsed) * 2))
            self.recorder.add_score(100 + time_bonus)
            if self.mazes_completed < self.total_mazes:
                self._maze_number += 1
                self._generate_maze(seed=self._maze_seed(self._maze_number))

    def draw(self, screen: pygame.Surface):
        W, H = self.SCREEN_WIDTH, self.SCREEN_HEIGHT

        # --- Labyrinthe ---
        for row in range(self.ROWS):
            for col in range(self.COLS):
                rx = self.offset_x + col * self.CELL_SIZE
                ry = self.offset_y + row * self.CELL_SIZE
                if self.grid[row][col]:
                    pygame.draw.rect(
                        screen, C_WALL, (rx, ry, self.CELL_SIZE, self.CELL_SIZE)
                    )
                else:
                    cell = (col, row)
                    color = C_VISITED if cell in self._visited_cells else C_PATH
                    pygame.draw.rect(
                        screen, color, (rx, ry, self.CELL_SIZE, self.CELL_SIZE)
                    )

        # --- Sortie ---
        ex, ey = self.exit_cell
        exit_rect = pygame.Rect(
            self.offset_x + ex * self.CELL_SIZE + 4,
            self.offset_y + ey * self.CELL_SIZE + 4,
            self.CELL_SIZE - 8,
            self.CELL_SIZE - 8,
        )
        t = time.time()
        pulse = int(180 + 75 * abs((t % 1.0) - 0.5) * 2)
        pygame.draw.rect(screen, (pulse, pulse, 50), exit_rect, border_radius=4)

        # --- Trail joueur ---
        for i, (tx, ty) in enumerate(self._trail):
            alpha = int(255 * i / len(self._trail))
            r = max(1, int(4 * i / len(self._trail)))
            pygame.draw.circle(screen, (*C_TRAIL, alpha), (tx, ty), r)

        # --- Joueur ---
        pygame.draw.circle(
            screen, C_PLAYER, (int(self.player_x), int(self.player_y)), 10
        )
        pygame.draw.circle(
            screen, (255, 255, 255), (int(self.player_x), int(self.player_y)), 10, 2
        )

        # --- Barre de progression (labyrinthes) ---
        font = pygame.font.SysFont("monospace", 16)
        for i in range(self.total_mazes):
            color = (80, 200, 80) if i < self.mazes_completed else (60, 60, 80)
            pygame.draw.rect(
                screen, color, (W // 2 - 60 + i * 45, H - 38, 35, 12), border_radius=3
            )

        label = font.render(
            f"Labyrinthes : {self.mazes_completed}/{self.total_mazes}",
            True,
            (150, 150, 150),
        )
        screen.blit(label, (W // 2 - label.get_width() // 2, H - 56))

        # --- Timer bar ---
        progress = 1.0 - (self.time_elapsed / self.GAME_DURATION)
        bar_w = int(W * progress)
        pygame.draw.rect(screen, (40, 40, 60), (0, H - 18, W, 18))
        c = (80, 200, 80) if progress > 0.3 else (200, 80, 80)
        pygame.draw.rect(screen, c, (0, H - 18, bar_w, 18))

        # --- Stats live ---
        font_s = pygame.font.SysFont("monospace", 15)
        stats = [
            f"Distance  : {self.total_distance/100:.1f}u",
            f"Demi-tours: {self.direction_changes}",
            f"Hésit.    : {self.hesitation_frames}f",
            f"Backtrack : {self.backtrack_count}",
        ]
        for i, s in enumerate(stats):
            surf = font_s.render(s, True, (120, 120, 140))
            screen.blit(surf, (W - 190, 10 + i * 20))

        # Hint clavier
        hint = font_s.render("Flèches = déplacement", True, (50, 50, 70))
        screen.blit(hint, (W // 2 - hint.get_width() // 2, H - 58))

    def is_over(self) -> bool:
        return (
            self.time_elapsed >= self.GAME_DURATION
            or self.mazes_completed >= self.total_mazes
        )

    def on_game_over(self):
        # Injecter les stats de navigation dans le recorder pour les features
        self.recorder.nav_distance = self.total_distance
        self.recorder.nav_direction_changes = self.direction_changes
        self.recorder.nav_hesitation = self.hesitation_frames
        self.recorder.nav_backtrack = self.backtrack_count
        self.recorder.mazes_completed = self.mazes_completed
