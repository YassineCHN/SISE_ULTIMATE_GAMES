"""
TwinStickShooter — Jeu de tir twin-stick
Joystick gauche  = déplacement du vaisseau
Joystick droit   = direction de tir (ou souris en fallback clavier)
Gâchette droite  = tirer (ou ESPACE)
Gâchette gauche  = dash/boost (ou LSHIFT)
Boutons          = bombes / capacités spéciales

Features capturées :
  - Précision du tir (% ennemis touchés)
  - Agressivité (fréquence de tir)
  - Mobilité (vitesse moyenne, distance parcourue)
  - Indépendance stick gauche/droit (corrélation mouvements)
  - Utilisation du dash (gâchette gauche)
  - Gestion des bombes (bouton Y)
"""

import pygame
import random
import math
import time
from dataclasses import dataclass
from typing import List
from games.base_game import BaseGame
from core.controller import ControllerState

# Couleurs
C_BG = (8, 8, 20)
C_PLAYER = (80, 200, 255)
C_BULLET = (255, 220, 80)
C_ENEMY = (220, 60, 60)
C_ENEMY2 = (220, 120, 40)
C_BOSS = (200, 40, 200)
C_PARTICLE = (255, 150, 50)
C_STAR = (200, 200, 255)
C_DASH = (100, 200, 255)
C_BOMB = (255, 80, 200)
C_HIT = (255, 255, 255)
C_SHIELD = (60, 160, 255)


@dataclass
class Entity:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: int = 10
    hp: int = 1
    alive: bool = True


@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float  # 0.0 → 1.0
    color: tuple
    size: int = 3


class TwinStickShooter(BaseGame):
    """
    Vagues d'ennemis à éliminer. 60 secondes.
    Sollicite : joystick gauche, joystick droit, gâchettes, boutons.
    """

    GAME_DURATION = 60
    PLAYER_SPEED = 220  # px/s
    BULLET_SPEED = 500
    DASH_SPEED = 600
    DASH_DURATION = 0.12  # secondes
    DASH_COOLDOWN = 1.0
    SHOOT_COOLDOWN = 0.12  # secondes entre tirs
    BOMB_COUNT = 3  # Bombes disponibles

    BG_COLOR = C_BG

    @property
    def game_id(self) -> str:
        return "shooter"

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def setup(self):
        W, H = self.SCREEN_WIDTH, self.SCREEN_HEIGHT
        self.time_elapsed = 0.0
        self.wave = 1
        self.wave_timer = 0.0
        self.spawn_timer = 0.0

        # Joueur
        self.player = Entity(x=W / 2, y=H * 0.75, radius=12, hp=5)
        self.invincible_timer = 0.0  # Frames d'invincibilité après coup

        # Entités
        self.bullets: List[Entity] = []
        self.enemies: List[Entity] = []
        self.particles: List[Particle] = []

        # Dash
        self.dash_timer = 0.0
        self.dash_cooldown = 0.0
        self.dashing = False
        self.dash_dir = (0.0, 0.0)

        # Tir
        self.shoot_cooldown = 0.0
        self.bombs_left = self.BOMB_COUNT
        self.bomb_cooldown = 0.0

        # Étoiles de fond
        self.stars = [
            (
                random.randint(0, W),
                random.randint(0, H),
                random.uniform(0.5, 2.0),
                random.randint(100, 220),
            )
            for _ in range(80)
        ]

        # Stats pour clustering
        self.shots_fired = 0
        self.shots_hit = 0
        self.enemies_killed = 0
        self.dash_uses = 0
        self.bomb_uses = 0
        self.total_dist = 0.0
        self._prev_pos = (W / 2, H * 0.75)
        self._prev_buttons = {}

        # Spawn première vague
        self._spawn_wave()

    # ------------------------------------------------------------------
    # Spawn
    # ------------------------------------------------------------------
    def _spawn_wave(self):
        W, H = self.SCREEN_WIDTH, self.SCREEN_HEIGHT
        n = 4 + self.wave * 2
        for _ in range(n):
            side = random.choice(["top", "left", "right"])
            if side == "top":
                x, y = random.randint(50, W - 50), -20
            elif side == "left":
                x, y = -20, random.randint(50, H // 2)
            else:
                x, y = W + 20, random.randint(50, H // 2)
            hp = 1 + self.wave // 3
            spd = random.uniform(60, 100 + self.wave * 10)
            ang = math.atan2(H / 2 - y, W / 2 - x) + random.uniform(-0.4, 0.4)
            color = C_BOSS if hp >= 3 else (C_ENEMY2 if hp == 2 else C_ENEMY)
            e = Entity(
                x=x,
                y=y,
                vx=math.cos(ang) * spd,
                vy=math.sin(ang) * spd,
                radius=8 + hp * 3,
                hp=hp,
            )
            self.enemies.append(e)

    def _spawn_particles(self, x, y, color, n=8):
        for _ in range(n):
            ang = random.uniform(0, math.pi * 2)
            spd = random.uniform(50, 200)
            self.particles.append(
                Particle(
                    x=x,
                    y=y,
                    vx=math.cos(ang) * spd,
                    vy=math.sin(ang) * spd,
                    life=1.0,
                    color=color,
                    size=random.randint(2, 5),
                )
            )

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, state: ControllerState, dt: float):
        self.time_elapsed += dt
        self.shoot_cooldown = max(0.0, self.shoot_cooldown - dt)
        self.dash_cooldown = max(0.0, self.dash_cooldown - dt)
        self.bomb_cooldown = max(0.0, self.bomb_cooldown - dt)
        self.invincible_timer = max(0.0, self.invincible_timer - dt)

        W, H = self.SCREEN_WIDTH, self.SCREEN_HEIGHT

        # --- Input mouvement (joystick gauche) ---
        move_x = state.axis_left_x
        move_y = state.axis_left_y

        # --- Input visée (joystick droit) ---
        aim_x = state.axis_right_x
        aim_y = state.axis_right_y

        # Fallback clavier : WASD = mouvement, flèches = visée
        keys = pygame.key.get_pressed()
        if abs(move_x) < 0.05 and abs(move_y) < 0.05:
            move_x = (1.0 if keys[pygame.K_d] else 0.0) - (
                1.0 if keys[pygame.K_q] else 0.0
            )
            move_y = (1.0 if keys[pygame.K_s] else 0.0) - (
                1.0 if keys[pygame.K_z] else 0.0
            )
        if abs(aim_x) < 0.05 and abs(aim_y) < 0.05:
            aim_x = (1.0 if keys[pygame.K_RIGHT] else 0.0) - (
                1.0 if keys[pygame.K_LEFT] else 0.0
            )
            aim_y = (1.0 if keys[pygame.K_DOWN] else 0.0) - (
                1.0 if keys[pygame.K_UP] else 0.0
            )

        # Normaliser mouvement
        mag = (move_x**2 + move_y**2) ** 0.5
        if mag > 1.0:
            move_x /= mag
            move_y /= mag

        # --- Dash (gâchette gauche ou LSHIFT) ---
        dash_input = state.button_l1 or state.trigger_left > 0.5 or keys[pygame.K_LSHIFT]
        if dash_input and self.dash_cooldown <= 0 and not self.dashing:
            if abs(move_x) > 0.1 or abs(move_y) > 0.1:
                self.dashing = True
                self.dash_timer = self.DASH_DURATION
                self.dash_dir = (move_x, move_y)
                self.dash_cooldown = self.DASH_COOLDOWN
                self.dash_uses += 1
                self._spawn_particles(self.player.x, self.player.y, C_DASH, 6)

        # --- Mouvement joueur ---
        if self.dashing:
            self.dash_timer -= dt
            dx = self.dash_dir[0] * self.DASH_SPEED * dt
            dy = self.dash_dir[1] * self.DASH_SPEED * dt
            if self.dash_timer <= 0:
                self.dashing = False
        else:
            dx = move_x * self.PLAYER_SPEED * dt
            dy = move_y * self.PLAYER_SPEED * dt

        new_x = max(self.player.radius, min(W - self.player.radius, self.player.x + dx))
        new_y = max(self.player.radius, min(H - self.player.radius, self.player.y + dy))
        dist = ((new_x - self.player.x) ** 2 + (new_y - self.player.y) ** 2) ** 0.5
        self.total_dist += dist
        self.player.x, self.player.y = new_x, new_y

        # --- Tir (gâchette droite ou ESPACE) ---
        shoot_input = state.button_r1 or state.trigger_right > 0.3 or keys[pygame.K_SPACE]
        if shoot_input and self.shoot_cooldown <= 0:
            # Direction de tir
            if abs(aim_x) > 0.1 or abs(aim_y) > 0.1:
                tx, ty = aim_x, aim_y
            else:
                # Auto-aim vers l'ennemi le plus proche
                tx, ty = 0.0, -1.0
                best_dist = float("inf")
                for e in self.enemies:
                    d = ((e.x - self.player.x) ** 2 + (e.y - self.player.y) ** 2) ** 0.5
                    if d < best_dist:
                        best_dist = d
                        tx = e.x - self.player.x
                        ty = e.y - self.player.y
                # Normaliser
                mag2 = (tx**2 + ty**2) ** 0.5
                if mag2 > 0:
                    tx /= mag2
                    ty /= mag2

            self.bullets.append(
                Entity(
                    x=self.player.x,
                    y=self.player.y,
                    vx=tx * self.BULLET_SPEED,
                    vy=ty * self.BULLET_SPEED,
                    radius=5,
                )
            )
            self.shots_fired += 1
            self.shoot_cooldown = self.SHOOT_COOLDOWN

        # --- Bombe (bouton Y=3 ou B=1 ou K) ---
        btn_bomb = state.buttons.get(3, False) or keys[pygame.K_k]
        was_bomb = self._prev_buttons.get("bomb", False)
        if (
            btn_bomb
            and not was_bomb
            and self.bombs_left > 0
            and self.bomb_cooldown <= 0
        ):
            self._detonate_bomb()
        self._prev_buttons["bomb"] = btn_bomb

        # --- Mise à jour balles ---
        for b in self.bullets:
            b.x += b.vx * dt
            b.y += b.vy * dt
            if b.x < 0 or b.x > W or b.y < 0 or b.y > H:
                b.alive = False

        # --- Mise à jour ennemis ---
        for e in self.enemies:
            e.x += e.vx * dt
            e.y += e.vy * dt
            # Légère poursuite du joueur
            ang = math.atan2(self.player.y - e.y, self.player.x - e.x)
            spd = (e.vx**2 + e.vy**2) ** 0.5
            e.vx = e.vx * 0.95 + math.cos(ang) * spd * 0.05
            e.vy = e.vy * 0.95 + math.sin(ang) * spd * 0.05
            # Hors écran → rebondir
            if e.x < -50 or e.x > W + 50 or e.y < -50 or e.y > H + 50:
                e.alive = False

        # --- Collisions balles / ennemis ---
        for b in self.bullets:
            if not b.alive:
                continue
            for e in self.enemies:
                if not e.alive:
                    continue
                d = ((b.x - e.x) ** 2 + (b.y - e.y) ** 2) ** 0.5
                if d < b.radius + e.radius:
                    b.alive = False
                    e.hp -= 1
                    self.shots_hit += 1
                    self._spawn_particles(e.x, e.y, C_HIT, 4)
                    if e.hp <= 0:
                        e.alive = False
                        self.enemies_killed += 1
                        self.recorder.add_score(10 * self.wave)
                        self._spawn_particles(e.x, e.y, C_PARTICLE, 12)

        # --- Collisions ennemis / joueur ---
        if self.invincible_timer <= 0:
            for e in self.enemies:
                if not e.alive:
                    continue
                d = ((e.x - self.player.x) ** 2 + (e.y - self.player.y) ** 2) ** 0.5
                if d < e.radius + self.player.radius:
                    self.player.hp -= 1
                    self.invincible_timer = 1.5
                    self._spawn_particles(self.player.x, self.player.y, C_SHIELD, 10)
                    e.alive = False

        # --- Mise à jour particules ---
        for p in self.particles:
            p.x += p.vx * dt
            p.y += p.vy * dt
            p.vx *= 0.92
            p.vy *= 0.92
            p.life -= dt * 1.5

        # --- Nettoyage ---
        self.bullets = [b for b in self.bullets if b.alive]
        self.enemies = [e for e in self.enemies if e.alive]
        self.particles = [p for p in self.particles if p.life > 0]

        # --- Nouvelle vague si tous les ennemis éliminés ---
        if not self.enemies:
            self.wave += 1
            self.recorder.add_score(50)
            self._spawn_wave()

    def _detonate_bomb(self):
        """Explose tous les ennemis à l'écran"""
        self.bombs_left -= 1
        self.bomb_cooldown = 2.0
        self.bomb_uses += 1
        for e in self.enemies:
            self.enemies_killed += 1
            self.shots_hit += 1
            self.recorder.add_score(5 * self.wave)
            self._spawn_particles(e.x, e.y, C_BOMB, 15)
            e.alive = False
        self._spawn_particles(self.SCREEN_WIDTH / 2, self.SCREEN_HEIGHT / 2, C_BOMB, 40)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------
    def draw(self, screen: pygame.Surface):
        W, H = self.SCREEN_WIDTH, self.SCREEN_HEIGHT

        # --- Étoiles ---
        for sx, sy, size, bright in self.stars:
            c = (bright, bright, bright)
            pygame.draw.circle(screen, c, (sx, sy), int(size))

        # --- Particules ---
        for p in self.particles:
            alpha = int(255 * p.life)
            r = max(1, int(p.size * p.life))
            c = tuple(min(255, int(ch * p.life)) for ch in p.color)
            pygame.draw.circle(screen, c, (int(p.x), int(p.y)), r)

        # --- Balles ---
        for b in self.bullets:
            pygame.draw.circle(screen, C_BULLET, (int(b.x), int(b.y)), b.radius)
            # Trainée
            pygame.draw.circle(
                screen,
                (180, 150, 30),
                (int(b.x - b.vx * 0.02), int(b.y - b.vy * 0.02)),
                b.radius - 2,
            )

        # --- Ennemis ---
        for e in self.enemies:
            color = C_BOSS if e.hp >= 3 else (C_ENEMY2 if e.hp == 2 else C_ENEMY)
            pygame.draw.circle(screen, color, (int(e.x), int(e.y)), e.radius)
            pygame.draw.circle(
                screen, (255, 255, 255), (int(e.x), int(e.y)), e.radius, 1
            )
            # Barre de vie si > 1 HP
            if e.hp > 1:
                bar_w = e.radius * 2
                filled = int(bar_w * e.hp / (1 + self.wave // 3))
                pygame.draw.rect(
                    screen,
                    (60, 60, 60),
                    (int(e.x) - e.radius, int(e.y) - e.radius - 8, bar_w, 5),
                )
                pygame.draw.rect(
                    screen,
                    (80, 220, 80),
                    (int(e.x) - e.radius, int(e.y) - e.radius - 8, filled, 5),
                )

        # --- Joueur ---
        if self.invincible_timer > 0 and int(self.time_elapsed * 10) % 2 == 0:
            pass  # Clignotement invincibilité
        else:
            col = C_DASH if self.dashing else C_PLAYER
            pygame.draw.circle(
                screen,
                col,
                (int(self.player.x), int(self.player.y)),
                self.player.radius,
            )
            pygame.draw.circle(
                screen,
                (255, 255, 255),
                (int(self.player.x), int(self.player.y)),
                self.player.radius,
                2,
            )

        # --- HP joueur ---
        for i in range(self.player.hp):
            pygame.draw.circle(screen, (80, 220, 80), (W - 20 - i * 22, 15), 7)
        for i in range(self.player.hp, 5):
            pygame.draw.circle(screen, (60, 60, 60), (W - 20 - i * 22, 15), 7)

        # --- Bombes ---
        font_s = pygame.font.SysFont("monospace", 15)
        bomb_txt = font_s.render(f"💣 x{self.bombs_left}  [K]", True, C_BOMB)
        screen.blit(bomb_txt, (W - 120, 30))

        # --- Wave + Stats live ---
        font_s2 = pygame.font.SysFont("monospace", 15)
        precision = int(self.shots_hit / max(1, self.shots_fired) * 100)
        stats = [
            f"Vague     : {self.wave}",
            f"Kills     : {self.enemies_killed}",
            f"Précision : {precision}%",
            f"Dash uses : {self.dash_uses}",
        ]
        for i, s in enumerate(stats):
            surf = font_s2.render(s, True, (120, 120, 150))
            screen.blit(surf, (W - 190, H - 100 + i * 20))

        # --- Timer bar ---
        progress = 1.0 - (self.time_elapsed / self.GAME_DURATION)
        bar_w = int(W * progress)
        pygame.draw.rect(screen, (30, 30, 50), (0, H - 12, W, 12))
        c = (80, 200, 80) if progress > 0.3 else (200, 80, 80)
        pygame.draw.rect(screen, c, (0, H - 12, bar_w, 12))

        # --- Dash cooldown ---
        if self.dash_cooldown > 0:
            cd_w = int(80 * (1 - self.dash_cooldown / self.DASH_COOLDOWN))
            pygame.draw.rect(screen, (40, 40, 80), (10, H - 28, 80, 8))
            pygame.draw.rect(screen, C_DASH, (10, H - 28, cd_w, 8))
            dash_lbl = font_s.render("DASH", True, (80, 120, 200))
            screen.blit(dash_lbl, (10, H - 44))

        # --- Instructions ---
        hint = font_s.render(
            "WASD/move  Flèches/aim  SPACE/shoot  SHIFT/dash  K/bomb",
            True,
            (40, 40, 60),
        )
        screen.blit(hint, (W // 2 - hint.get_width() // 2, H - 28))

    # ------------------------------------------------------------------
    # is_over / on_game_over
    # ------------------------------------------------------------------
    def is_over(self) -> bool:
        return self.time_elapsed >= self.GAME_DURATION or self.player.hp <= 0

    def on_game_over(self):
        precision = self.shots_hit / max(1, self.shots_fired)
        self.recorder.shooter_precision = precision
        self.recorder.shooter_kills = self.enemies_killed
        self.recorder.shooter_dash_uses = self.dash_uses
        self.recorder.shooter_bomb_uses = self.bomb_uses
        self.recorder.shooter_distance = self.total_dist
        self.recorder.shooter_wave = self.wave
