"""
RacingGame — Course top-down vue du dessus
Circuit fermé, 3 tours à compléter.

Contrôles clavier : ↑ accélérer  ↓ freiner  ←/→ tourner  SHIFT nitro
Manette           : gâchette droite/gauche, joystick gauche X, bouton A

Features capturées :
  - Douceur du volant (std steering)
  - Agressivité accélération/freinage
  - Utilisation nitro
  - Régularité des tours (std lap times)
  - Sorties de piste
  - Vitesse moyenne
"""

import pygame, math, random, time
from games.base_game import BaseGame
from core.controller import ControllerState

# ── Palette ───────────────────────────────────────────────────────
C_GRASS = (34, 120, 34)
C_GRASS_D = (24, 90, 24)
C_ROAD = (80, 80, 88)
C_ROAD_EDGE = (60, 60, 68)
C_KERB_R = (210, 30, 30)
C_KERB_W = (240, 240, 240)
C_LINE = (240, 240, 240)
C_CAR = (60, 160, 255)
C_CAR_DARK = (30, 80, 140)
C_NITRO_C = (255, 180, 30)
C_HUD = (220, 220, 220)
C_OFFROAD = (200, 80, 20)
C_CHECKPOINT = (255, 220, 50)

LAPS_WIN = 3
ROAD_W = 38  # largeur de la piste en pixels
KERB_W = 6  # largeur des bords colorés


# ── Circuit : liste de waypoints (x, y) ──────────────────────────
def build_circuit(W, H):
    """Circuit ovale avec 2 chicanes douces, centré et dans les limites écran."""
    cx, cy = W // 2, H // 2 + 15
    rx, ry = W * 0.30, H * 0.30  # marges généreuses sur tous les bords

    N = 120
    pts = []
    for i in range(N):
        t = i / N * math.tau
        x = cx + rx * math.cos(t)
        y = cy + ry * math.sin(t)
        # Chicane en haut (vers l'intérieur)
        if 0.9 < t < 1.5:
            blend = math.sin((t - 0.9) / (1.5 - 0.9) * math.pi)
            y -= 30 * blend
        # Chicane en bas (vers l'intérieur)
        if 4.0 < t < 4.7:
            blend = math.sin((t - 4.0) / (4.7 - 4.0) * math.pi)
            y += 30 * blend
        pts.append((x, y))
    return pts


class RacingGame(BaseGame):

    BG_COLOR = C_GRASS
    GAME_DURATION = 120
    CAR_SPEED_MAX = 280  # px/s
    CAR_ACCEL = 140
    CAR_BRAKE = 220
    CAR_DECEL = 80
    TURN_SPEED = 2.8  # rad/s
    NITRO_MULT = 1.6
    NITRO_CHARGES = 3

    @property
    def game_id(self):
        return "racing"

    # ── Setup ─────────────────────────────────────────────────────
    def setup(self):
        W, H = self.SCREEN_WIDTH, self.SCREEN_HEIGHT
        self.circuit = build_circuit(W, H)
        self.n_pts = len(self.circuit)

        # Position de départ : premier waypoint du circuit
        sx, sy = self.circuit[0]
        sx2, sy2 = self.circuit[1]
        start_angle = math.atan2(sy2 - sy, sx2 - sx)

        self.car_x = float(sx)
        self.car_y = float(sy)
        self.car_angle = start_angle
        self.car_speed = 0.0
        self.steering = 0.0  # volant lissé

        self.time_elapsed = 0.0
        self.lap = 0
        self.lap_start = 0.0
        self.lap_times = []
        self.nitro_left = self.NITRO_CHARGES
        self.nitro_on = False
        self.nitro_timer = 0.0
        self.nitro_cd = 0.0
        self.nitro_uses = 0
        self.off_frames = 0
        self.off_events = 0
        self._off_t = 0.0
        self._prev_btns = {}
        self.finish_flash = 0.0

        # Checkpoints : index du prochain waypoint à atteindre
        self._next_wp = 1
        self._wp_radius = 28

        # Stats
        self.steer_s = []
        self.speed_s = []
        self.accel_s = []
        self.brake_s = []

    # ── Helpers circuit ───────────────────────────────────────────
    def _nearest_wp(self):
        """Retourne l'index du waypoint le plus proche de la voiture"""
        best, best_d = 0, float("inf")
        for i, (px, py) in enumerate(self.circuit):
            d = (self.car_x - px) ** 2 + (self.car_y - py) ** 2
            if d < best_d:
                best, best_d = i, d
        return best

    def _on_track(self):
        """Vérifie si la voiture est sur la piste (proche d'un waypoint)"""
        nearest = self._nearest_wp()
        px, py = self.circuit[nearest]
        dist = math.hypot(self.car_x - px, self.car_y - py)
        return dist < ROAD_W * 1.2

    # ── Update ────────────────────────────────────────────────────
    def update(self, state: ControllerState, dt: float):
        self.time_elapsed += dt
        self.nitro_timer = max(0.0, self.nitro_timer - dt)
        self.nitro_cd = max(0.0, self.nitro_cd - dt)
        self.finish_flash = max(0.0, self.finish_flash - dt * 2)
        if self.nitro_timer <= 0:
            self.nitro_on = False

        keys = pygame.key.get_pressed()

        # Inputs
        steer = state.axis_left_x
        if abs(steer) < 0.05 and state.source == "keyboard":
            steer = (1.0 if keys[pygame.K_RIGHT] else 0.0) - (
                1.0 if keys[pygame.K_LEFT] else 0.0
            )

        accel = 1.0 if (state.button_r1 or state.trigger_right > 0.1) else 0.0
        if accel < 0.05:
            accel = 1.0 if keys[pygame.K_UP] else 0.0

        brake = 1.0 if (state.button_l1 or state.trigger_left > 0.1) else 0.0
        if brake < 0.05:
            brake = 1.0 if keys[pygame.K_DOWN] else 0.0

        # Nitro
        nitro_btn = state.buttons.get(0, False) or keys[pygame.K_LSHIFT]
        was_n = self._prev_btns.get("n", False)
        if nitro_btn and not was_n and self.nitro_left > 0 and self.nitro_cd <= 0:
            self.nitro_on = True
            self.nitro_timer = 2.5
            self.nitro_left -= 1
            self.nitro_cd = 4.0
            self.nitro_uses += 1
        self._prev_btns["n"] = nitro_btn

        # Physique
        max_spd = self.CAR_SPEED_MAX * (self.NITRO_MULT if self.nitro_on else 1.0)
        max_rev = self.CAR_SPEED_MAX * 0.4  # vitesse max en marche arrière

        if accel > 0.05:
            self.car_speed += self.CAR_ACCEL * accel * dt
        else:
            # Décélération naturelle vers 0
            if self.car_speed > 0:
                self.car_speed -= self.CAR_DECEL * dt
                self.car_speed = max(0.0, self.car_speed)
            elif self.car_speed < 0:
                self.car_speed += self.CAR_DECEL * dt
                self.car_speed = min(0.0, self.car_speed)

        if brake > 0.05:
            self.car_speed -= self.CAR_BRAKE * brake * dt

        # Hors piste → ralentissement
        on_track = self._on_track()
        if not on_track:
            self.car_speed *= 1 - 2.5 * dt
            self.off_frames += 1
            self._off_t += dt
            if self._off_t > 0.5:
                self.off_events += 1
                self._off_t = 0.0
        else:
            self._off_t = 0.0

        self.car_speed = max(-max_rev, min(max_spd, self.car_speed))

        # Rotation (proportionnelle à la vitesse)
        self.steering += (steer - self.steering) * min(1.0, dt * 8)
        spd_ratio = self.car_speed / self.CAR_SPEED_MAX
        self.car_angle += self.steering * self.TURN_SPEED * spd_ratio * dt

        # Déplacement
        self.car_x += math.cos(self.car_angle) * self.car_speed * dt
        self.car_y += math.sin(self.car_angle) * self.car_speed * dt

        # Garder dans les limites de l'écran
        W, H = self.SCREEN_WIDTH, self.SCREEN_HEIGHT
        self.car_x = max(10.0, min(W - 10.0, self.car_x))
        self.car_y = max(10.0, min(H - 10.0, self.car_y))

        # Checkpoint / Lap
        nx, ny = self.circuit[self._next_wp]
        if math.hypot(self.car_x - nx, self.car_y - ny) < self._wp_radius:
            self._next_wp = (self._next_wp + 1) % self.n_pts
            if self._next_wp == 1:  # Tour complet (on repasse par wp 0→1)
                self.lap += 1
                lt = self.time_elapsed - self.lap_start
                self.lap_times.append(lt)
                self.lap_start = self.time_elapsed
                self.finish_flash = 1.0
                self.recorder.add_score(max(0, int(5000 - lt * 20)))

        # Samples stats
        self.steer_s.append(abs(self.steering))
        self.speed_s.append(self.car_speed)
        self.accel_s.append(accel)
        self.brake_s.append(brake)

    # ── Draw ──────────────────────────────────────────────────────
    def draw(self, screen: pygame.Surface):
        W, H = self.SCREEN_WIDTH, self.SCREEN_HEIGHT

        # ── Fond herbe ──
        screen.fill(C_GRASS)

        # ── Circuit : on dessine des bandes de route entre waypoints ──
        pts = self.circuit
        n = self.n_pts

        # 1. Kerb (bords colorés — dessiné en premier, plus large)
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            c = C_KERB_R if i % 2 == 0 else C_KERB_W
            pygame.draw.line(
                screen,
                c,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                ROAD_W * 2 + KERB_W * 2,
            )
        # Cercles aux jonctions pour combler les trous du kerb
        for i in range(n):
            p = pts[i]
            c = C_KERB_R if i % 2 == 0 else C_KERB_W
            pygame.draw.circle(screen, c, (int(p[0]), int(p[1])), ROAD_W + KERB_W)

        # 2. Route par dessus
        for i in range(n):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            pygame.draw.line(
                screen,
                C_ROAD,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                ROAD_W * 2,
            )
        # Cercles aux jonctions pour combler les trous de la route
        for i in range(n):
            p = pts[i]
            pygame.draw.circle(screen, C_ROAD, (int(p[0]), int(p[1])), ROAD_W)

        # 3. Ligne centrale tiretée
        for i in range(0, n, 3):
            p1 = pts[i]
            p2 = pts[(i + 1) % n]
            pygame.draw.line(
                screen, C_LINE, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), 2
            )

        # 5. Ligne de départ/arrivée
        sp = pts[0]
        sp2 = pts[1]
        dx = sp2[0] - sp[0]
        dy = sp2[1] - sp[1]
        lg = math.hypot(dx, dy) or 1
        px = -dy / lg * (ROAD_W + 4)
        py = dx / lg * (ROAD_W + 4)
        pygame.draw.line(
            screen,
            C_CHECKPOINT,
            (int(sp[0] - px), int(sp[1] - py)),
            (int(sp[0] + px), int(sp[1] + py)),
            3,
        )

        # ── Prochain checkpoint (flèche) ──
        nx, ny = self.circuit[self._next_wp]
        pygame.draw.circle(screen, C_CHECKPOINT, (int(nx), int(ny)), 6, 2)

        # ── Voiture ──
        cw, cl = 12, 20  # demi-largeur, demi-longueur
        angle = self.car_angle
        cos_a, sin_a = math.cos(angle), math.sin(angle)

        def rot(lx, ly):
            return (
                self.car_x + lx * cos_a - ly * sin_a,
                self.car_y + lx * sin_a + ly * cos_a,
            )

        body = [rot(-cl, -cw), rot(cl, -cw), rot(cl, cw), rot(-cl, cw)]
        col = C_NITRO_C if self.nitro_on else C_CAR
        pygame.draw.polygon(screen, col, [(int(x), int(y)) for x, y in body])

        # Pare-brise
        win = [
            rot(cl * 0.2, -cw * 0.7),
            rot(cl * 0.8, -cw * 0.7),
            rot(cl * 0.8, cw * 0.7),
            rot(cl * 0.2, cw * 0.7),
        ]
        pygame.draw.polygon(screen, C_CAR_DARK, [(int(x), int(y)) for x, y in win])

        # Flammes nitro
        if self.nitro_on:
            for _ in range(3):
                fx, fy = rot(
                    -cl - random.randint(3, 12), random.uniform(-cw * 0.4, cw * 0.4)
                )
                pygame.draw.circle(
                    screen, C_NITRO_C, (int(fx), int(fy)), random.randint(3, 7)
                )

        # ── HUD ──
        fb = pygame.font.SysFont("monospace", 26, bold=True)
        fs = pygame.font.SysFont("monospace", 15)

        kmh = int(abs(self.car_speed) * 3.6)
        sc = C_NITRO_C if self.nitro_on else C_HUD
        rev_str = " R" if self.car_speed < -1 else ""
        ssurf = fb.render(f"{kmh} km/h{rev_str}", True, sc)
        screen.blit(ssurf, (W - ssurf.get_width() - 12, H - 48))

        lsurf = fb.render(f"Lap {min(self.lap+1, LAPS_WIN)}/{LAPS_WIN}", True, C_HUD)
        screen.blit(lsurf, (W // 2 - lsurf.get_width() // 2, 8))

        cur_t = self.time_elapsed - self.lap_start
        csurf = fs.render(f"{cur_t:.2f}s", True, (160, 160, 160))
        screen.blit(csurf, (W // 2 - csurf.get_width() // 2, 40))

        if self.lap_times:
            bsurf = fs.render(
                f"Best: {min(self.lap_times):.2f}s", True, (100, 220, 100)
            )
            screen.blit(bsurf, (W // 2 - bsurf.get_width() // 2, 58))

        # Nitro
        for i in range(self.nitro_left):
            pygame.draw.rect(
                screen, C_NITRO_C, (10 + i * 24, H - 28, 18, 10), border_radius=3
            )
        for i in range(self.nitro_left, self.NITRO_CHARGES):
            pygame.draw.rect(
                screen, (50, 40, 10), (10 + i * 24, H - 28, 18, 10), border_radius=3
            )
        screen.blit(fs.render("NITRO [SHIFT]", True, C_NITRO_C), (10, H - 46))

        # Flash tour
        if self.finish_flash > 0:
            ff = pygame.font.SysFont("monospace", 36, bold=True)
            fsurf = ff.render(
                "TOUR COMPLÉTÉ !", True, (int(255 * self.finish_flash), 255, 80)
            )
            screen.blit(fsurf, (W // 2 - fsurf.get_width() // 2, H // 2 - 50))

        # Hors piste
        if not self._on_track():
            wsurf = fs.render("⚠ HORS PISTE !", True, C_OFFROAD)
            screen.blit(wsurf, (W // 2 - wsurf.get_width() // 2, H // 2 + 20))

        # Timer bar
        prog = 1.0 - (self.time_elapsed / self.GAME_DURATION)
        pygame.draw.rect(screen, (20, 60, 20), (0, H - 10, W, 10))
        c = (80, 200, 80) if prog > 0.3 else (200, 80, 80)
        pygame.draw.rect(screen, c, (0, H - 10, int(W * prog), 10))

        # Hint
        hsurf = fs.render(
            "↑ accélérer  ↓ freiner  ←/→ tourner  SHIFT nitro", True, (20, 80, 20)
        )
        screen.blit(hsurf, (W // 2 - hsurf.get_width() // 2, H - 58))

    # ── is_over / on_game_over ────────────────────────────────────
    def is_over(self):
        return self.time_elapsed >= self.GAME_DURATION or self.lap >= LAPS_WIN

    def _mean(self, v):
        return sum(v) / len(v) if v else 0.0

    def _std(self, v):
        if len(v) < 2:
            return 0.0
        m = self._mean(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / len(v))

    def on_game_over(self):
        self.recorder.racing_steering_std = self._std(self.steer_s)
        self.recorder.racing_speed_mean = self._mean(self.speed_s)
        self.recorder.racing_throttle_mean = self._mean(self.accel_s)
        self.recorder.racing_brake_mean = self._mean(self.brake_s)
        self.recorder.racing_offroad = self.off_events
        self.recorder.racing_nitro_uses = self.nitro_uses
        self.recorder.racing_lap_std = self._std(self.lap_times)
        self.recorder.racing_laps_done = len(self.lap_times)
