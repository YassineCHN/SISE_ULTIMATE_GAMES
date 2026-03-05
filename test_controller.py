"""
test_controller.py — Outil de diagnostic manette PS4/PS5 en temps réel

Affiche :
  - Informations sur la manette détectée
  - Tous les axes bruts (avec leur index)
  - Tous les boutons bruts (avec leur index)
  - L'état hat/D-pad
  - Les valeurs interprétées par le module Controller (ControllerState)
  - Représentation visuelle des joysticks

Lancement : python test_controller.py  (ou uv run test_controller.py)
"""

import sys
import os
import pygame
import time

# Ajout du dossier racine au path pour importer core.*
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from core.controller import Controller

# ── Couleurs ────────────────────────────────────────────────────────────────
BLACK   = (10, 10, 20)
WHITE   = (240, 240, 240)
GRAY    = (120, 120, 140)
GREEN   = (80, 220, 80)
RED     = (220, 60, 60)
YELLOW  = (240, 220, 60)
CYAN    = (60, 210, 240)
ORANGE  = (240, 150, 40)
PURPLE  = (180, 80, 220)
BLUE    = (60, 120, 240)

# ── Dimensions ───────────────────────────────────────────────────────────────
W, H = 960, 700
FPS  = 60

# ── Helpers ──────────────────────────────────────────────────────────────────
def bar(surface, x, y, value, width=160, height=14, color=CYAN, label=""):
    """Barre horizontale pour visualiser une valeur [-1, 1] ou [0, 1]"""
    pygame.draw.rect(surface, (40, 40, 60), (x, y, width, height))
    # Détecter si valeur normalisée [0,1] ou bipolaire [-1,1]
    if value < -0.01:  # bipolaire
        cx = x + width // 2
        bar_w = int(abs(value) * (width // 2))
        pygame.draw.rect(surface, color, (cx - bar_w, y + 1, bar_w, height - 2))
        pygame.draw.line(surface, GRAY, (cx, y), (cx, y + height), 1)
    else:
        bar_w = int(value * width)
        pygame.draw.rect(surface, color, (x, y + 1, bar_w, height - 2))
    # Cadre
    pygame.draw.rect(surface, GRAY, (x, y, width, height), 1)


def joystick_circle(surface, cx, cy, radius, lx, ly, color=CYAN):
    """Cercle représentant la position d'un joystick"""
    pygame.draw.circle(surface, (30, 30, 50), (cx, cy), radius)
    pygame.draw.circle(surface, GRAY, (cx, cy), radius, 1)
    # Croix centrale
    pygame.draw.line(surface, (60, 60, 80), (cx - radius, cy), (cx + radius, cy), 1)
    pygame.draw.line(surface, (60, 60, 80), (cx, cy - radius), (cx, cy + radius), 1)
    # Point de position
    px = int(cx + lx * (radius - 6))
    py = int(cy + ly * (radius - 6))
    pygame.draw.circle(surface, color, (px, py), 8)
    pygame.draw.circle(surface, WHITE, (px, py), 8, 1)


def trigger_rect(surface, x, y, value, label, color=ORANGE):
    """Rectangle vertical pour gâchette [0, 1]"""
    h = 80
    w = 30
    pygame.draw.rect(surface, (30, 30, 50), (x, y, w, h))
    fill_h = int(value * h)
    pygame.draw.rect(surface, color, (x, y + h - fill_h, w, fill_h))
    pygame.draw.rect(surface, GRAY, (x, y, w, h), 1)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("SISE — Diagnostic Manette")
    clock = pygame.time.Clock()

    font_title = pygame.font.SysFont("monospace", 18, bold=True)
    font_main  = pygame.font.SysFont("monospace", 14)
    font_small = pygame.font.SysFont("monospace", 12)

    # Initialisation manette + Controller interprété
    controller = Controller()
    joystick = controller.joystick  # accès direct à pygame.joystick

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        screen.fill(BLACK)

        # ── En-tête ────────────────────────────────────────────────────────
        title = "SISE — DIAGNOSTIC MANETTE   [ESC pour quitter]"
        screen.blit(font_title.render(title, True, WHITE), (20, 12))
        pygame.draw.line(screen, GRAY, (0, 32), (W, 32), 1)

        # ── Infos manette ──────────────────────────────────────────────────
        y = 40
        if joystick:
            name = joystick.get_name()
            ctype = controller.controller_type
            n_axes = joystick.get_numaxes()
            n_btns = joystick.get_numbuttons()
            n_hats = joystick.get_numhats()
            info_color = GREEN
            screen.blit(font_main.render(f"Manette : {name}", True, GREEN), (20, y))
            screen.blit(font_main.render(
                f"Type detecte : {ctype.upper()}   axes={n_axes}  boutons={n_btns}  hats={n_hats}",
                True, CYAN), (20, y + 18))
            amap = controller._axis_map
            screen.blit(font_small.render(
                f"Mapping axes → lx:{amap['lx']} ly:{amap['ly']} rx:{amap['rx']} ry:{amap['ry']} lt:{amap['lt']} rt:{amap['rt']}",
                True, YELLOW), (20, y + 36))
        else:
            screen.blit(font_main.render("Aucune manette — MODE CLAVIER", True, RED), (20, y))
            screen.blit(font_small.render(
                "Fleches=joystick | Z/X/C/V=boutons | A=LT | E=RT", True, GRAY), (20, y + 18))

        pygame.draw.line(screen, (40, 40, 60), (0, y + 56), (W, y + 56), 1)

        # ── Lecture état interprété ────────────────────────────────────────
        state = controller.get_state()

        # ══ COLONNE GAUCHE — Axes bruts ════════════════════════════════════
        col1_x = 20
        y_axes = 115

        screen.blit(font_title.render("AXES BRUTS (pygame)", True, YELLOW), (col1_x, y_axes - 20))

        if joystick:
            for i in range(joystick.get_numaxes()):
                v = joystick.get_axis(i)
                # Identifier ce que cet axe représente
                hint = ""
                am = controller._axis_map
                if   i == am.get("lx"): hint = " ← lx (joystick G horizontale)"
                elif i == am.get("ly"): hint = " ← ly (joystick G verticale)"
                elif i == am.get("rx"): hint = " ← rx (joystick D horizontale)"
                elif i == am.get("ry"): hint = " ← ry (joystick D verticale)"
                elif i == am.get("lt"): hint = " ← L2 / LT"
                elif i == am.get("rt"): hint = " ← R2 / RT"

                label = f"Axe {i:2d} : {v:+.3f}"
                lbl_color = CYAN if hint else GRAY
                screen.blit(font_small.render(label + hint, True, lbl_color), (col1_x, y_axes))
                bar(screen, col1_x + 180, y_axes + 1, v, width=140, height=12,
                    color=CYAN if hint else (70, 70, 90))
                y_axes += 18
        else:
            screen.blit(font_small.render("(pas de manette)", True, GRAY), (col1_x, y_axes))

        # ══ COLONNE CENTRE — Boutons bruts ═════════════════════════════════
        col2_x = 400
        y_btns = 115
        screen.blit(font_title.render("BOUTONS BRUTS", True, YELLOW), (col2_x, y_btns - 20))

        if joystick:
            nb = joystick.get_numbuttons()
            cols = 2
            per_col = (nb + cols - 1) // cols
            for i in range(nb):
                pressed = joystick.get_button(i)
                col_off = (i // per_col) * 150
                row = i % per_col

                # Identifier l'usage
                l1i, r1i = controller.L1R1_MAP[controller.controller_type]
                hint = ""
                if   i == l1i: hint = " L1"
                elif i == r1i: hint = " R1"
                elif i == 0:   hint = " ×/A"
                elif i == 1:   hint = " ○/B"
                elif i == 2:   hint = " □/X"
                elif i == 3:   hint = " △/Y"

                color = GREEN if pressed else (50, 50, 70)
                pygame.draw.rect(screen, color, (col2_x + col_off, y_btns + row * 18, 12, 12))
                pygame.draw.rect(screen, GRAY, (col2_x + col_off, y_btns + row * 18, 12, 12), 1)
                label = f"Btn {i:2d}{hint}"
                screen.blit(font_small.render(label, True, WHITE if pressed else GRAY),
                            (col2_x + col_off + 16, y_btns + row * 18))
        else:
            screen.blit(font_small.render("(pas de manette)", True, GRAY), (col2_x, y_btns))

        # Hat / D-pad
        if joystick and joystick.get_numhats() > 0:
            hat = joystick.get_hat(0)
            hat_y = y_btns + (per_col if joystick else 0) * 18 + 8 if joystick else y_btns + 8
            screen.blit(font_small.render(f"Hat/D-pad : {hat}", True,
                         GREEN if hat != (0, 0) else GRAY), (col2_x, 520))

        # ══ COLONNE DROITE — État interprété ═══════════════════════════════
        col3_x = 700
        y_int = 95

        screen.blit(font_title.render("ETAT INTERPRETE", True, ORANGE), (col3_x, y_int - 20))

        interp_lines = [
            ("lx (stick G →)", f"{state.axis_left_x:+.3f}",  state.axis_left_x,  CYAN),
            ("ly (stick G ↓)", f"{state.axis_left_y:+.3f}",  state.axis_left_y,  CYAN),
            ("rx (stick D →)", f"{state.axis_right_x:+.3f}", state.axis_right_x, PURPLE),
            ("ry (stick D ↓)", f"{state.axis_right_y:+.3f}", state.axis_right_y, PURPLE),
            ("LT / L2",        f"{state.trigger_left:.3f}",  state.trigger_left, ORANGE),
            ("RT / R2",        f"{state.trigger_right:.3f}", state.trigger_right,ORANGE),
        ]
        for name, val_str, val, color in interp_lines:
            screen.blit(font_small.render(f"{name:<18} {val_str}", True, color), (col3_x, y_int))
            bar(screen, col3_x, y_int + 14, val, width=230, height=8, color=color)
            y_int += 28

        # L1 / R1
        y_int += 4
        l1_color = GREEN if state.button_l1 else RED
        r1_color = GREEN if state.button_r1 else RED
        screen.blit(font_small.render(
            f"L1 : {'APPUYE' if state.button_l1 else 'relache'}", True, l1_color), (col3_x, y_int))
        screen.blit(font_small.render(
            f"R1 : {'APPUYE' if state.button_r1 else 'relache'}", True, r1_color), (col3_x, y_int + 18))

        # Source
        src_color = CYAN if state.source == "controller" else YELLOW
        screen.blit(font_small.render(f"Source : {state.source}", True, src_color), (col3_x, y_int + 40))

        # ══ SECTION BASSE — Visualisation joysticks ════════════════════════
        pygame.draw.line(screen, (40, 40, 60), (0, 540), (W, 540), 1)
        screen.blit(font_title.render("VISUALISATION JOYSTICKS", True, WHITE), (20, 548))

        R = 55  # rayon des cercles

        # Stick gauche
        screen.blit(font_small.render("Joystick G", True, CYAN), (90, 565))
        joystick_circle(screen, 120, 635, R, state.axis_left_x, state.axis_left_y, CYAN)
        screen.blit(font_small.render(
            f"({state.axis_left_x:+.2f}, {state.axis_left_y:+.2f})", True, CYAN), (75, 700 - 10))

        # Stick droit
        screen.blit(font_small.render("Joystick D", True, PURPLE), (280, 565))
        joystick_circle(screen, 310, 635, R, state.axis_right_x, state.axis_right_y, PURPLE)
        screen.blit(font_small.render(
            f"({state.axis_right_x:+.2f}, {state.axis_right_y:+.2f})", True, PURPLE), (265, 700 - 10))

        # Gâchettes
        screen.blit(font_small.render("L2", True, ORANGE), (490, 565))
        trigger_rect(screen, 490, 580, state.trigger_left, "L2", ORANGE)
        screen.blit(font_small.render(f"{state.trigger_left:.2f}", True, ORANGE), (485, 665))

        screen.blit(font_small.render("R2", True, ORANGE), (540, 565))
        trigger_rect(screen, 540, 580, state.trigger_right, "R2", RED)
        screen.blit(font_small.render(f"{state.trigger_right:.2f}", True, RED), (535, 665))

        # D-pad
        hat = state.hat
        screen.blit(font_small.render("D-pad", True, WHITE), (625, 565))
        hat_cx, hat_cy = 645, 625
        hat_r = 28
        pygame.draw.rect(screen, (30, 30, 50), (hat_cx - hat_r, hat_cy - hat_r, hat_r * 2, hat_r * 2))
        pygame.draw.rect(screen, GRAY, (hat_cx - hat_r, hat_cy - hat_r, hat_r * 2, hat_r * 2), 1)
        if hat != (0, 0):
            dx, dy = hat
            px = hat_cx + dx * (hat_r - 8)
            py = hat_cy - dy * (hat_r - 8)  # pygame hat Y est inversé
            pygame.draw.circle(screen, WHITE, (px, py), 6)

        # Boutons principaux interprétés
        main_btns = {
            0: ("×/A", (200, 200)),
            1: ("○/B", (240, 200)),
            2: ("□/X", (200, 160)),  # noqa (positional approximation)
            3: ("△/Y", (240, 160)),
        }
        screen.blit(font_small.render("Boutons", True, WHITE), (740, 565))
        for idx, (label, (bx_off, by_off)) in main_btns.items():
            pressed = state.buttons.get(idx, False)
            bx = 740 + (idx % 2) * 55
            by = 590 + (idx // 2) * 35
            color = GREEN if pressed else (40, 40, 60)
            pygame.draw.circle(screen, color, (bx + 12, by + 12), 14)
            pygame.draw.circle(screen, GRAY, (bx + 12, by + 12), 14, 1)
            screen.blit(font_small.render(label, True, WHITE if pressed else GRAY), (bx + 5, by + 7))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
