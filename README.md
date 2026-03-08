# 🎮 SISE Ultimate Games — Controller Profiler

> **Projet IA Temps Réel · Master SISE 2025–2026**  
> Analyse comportementale de joueurs via manette Xbox/PS en temps réel, clustering de profils et coaching IA.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dashboard-Dash%2FPlotly-informational?logo=plotly)](https://dash.plotly.com/)
[![Supabase](https://img.shields.io/badge/Backend-Supabase-3ECF8E?logo=supabase)](https://supabase.com/)
[![Mistral](https://img.shields.io/badge/LLM-Mistral%20AI-orange)](https://mistral.ai/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)
[![Présentation](https://img.shields.io/badge/Slides-Canva-8B3DFF?logo=canva)](https://www.canva.com/design/DAHC_9YOXjQ/RTeTzMMWfSxd0moCdxwVgA/edit?utm_content=DAHC_9YOXjQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

---

## 📖 Présentation

**SISE Ultimate Games** est une application de **profilage de joueurs en temps réel** à partir des inputs d'une manette de jeu (Xbox / PlayStation). Chaque pression de bouton, mouvement de joystick et utilisation de gâchette est capturé à 30 Hz, analysé et transformé en un profil comportemental unique.

Le système comprend :
- **4 mini-jeux Pygame** conçus pour solliciter différentes dimensions de gameplay
- **Un pipeline ML** (UMAP + K-Means) pour clusteriser les joueurs en profils
- **Un coach IA** (Mistral) générant des analyses personnalisées post-session
- **Un agent imitateur** capable de rejouer les patterns d'un joueur réel
- **Un dashboard Dash** temps réel centralisant toutes les données

---

## ✨ Fonctionnalités

| Fonctionnalité | Description |
|---|---|
| 🕹️ **4 mini-jeux** | Reflex, Labyrinthe, Shooter, Racing — chacun mesure des axes différents |
| 📡 **Capture temps réel** | Inputs manette échantillonnés à 30 Hz via Pygame |
| 🧠 **Clustering UMAP + K-Means** | 3–4 profils de joueurs : Agressif, Prudent, Précis, Chaotique |
| 🤖 **Agent imitateur** | Rejoue fidèlement le style d'un joueur enregistré (bruit configurable) |
| 💬 **Coach IA Mistral** | Analyse de session personnalisée avec conseils actionnables |
| 📊 **Dashboard Dash** | Leaderboard, résumés IA, visualisation UMAP, flux d'inputs live |
| ☁️ **Sync Supabase** | Persistance des sessions et résumés dans le cloud |

---

## 🏗️ Architecture

```
SISE_ULTIMATE_GAMES/
│
├── main.py                  # Point d'entrée — lancement des jeux
├── requirements.txt         # Dépendances Python
├── pyproject.toml
├── .env                     # Variables d'environnement (à créer, voir ci-dessous)
│
├── games/                   # Mini-jeux Pygame
│   ├── base_game.py         # Classe abstraite commune
│   ├── reflex_game.py       # Jeu de réflexes (boutons)
│   ├── labyrinth_game.py    # Labyrinthe (joystick gauche)
│   ├── shooter_game.py      # Shooter (joysticks + gâchettes)
│   └── racing_game.py       # Racing (joystick + nitro)
│
├── core/                    # Moteur analytique
│   ├── controller.py        # Lecture manette / clavier (abstraction unifiée)
│   ├── recorder.py          # Enregistrement session + extraction de features
│   ├── llm_summary.py       # Intégration Mistral AI (résumés + chat)
│   ├── agent.py             # Agent imitateur IA
│   └── supabase_client.py   # Client Supabase (persistance cloud)
│
├── app/                     # Dashboard Dash
│   ├── app.py               # Application principale multi-pages
│   └── assets/              # CSS, polices, images
│
└── test_controller.py       # Diagnostic manette (standalone)
```

---

## ⚙️ Prérequis

- **Python 3.10+**
- **Une manette Xbox ou PlayStation** connectée en USB (ou via Bluetooth)  
  *(un fallback clavier est disponible mais les features seront moins riches)*
- Comptes **Supabase** et **Mistral AI** (gratuits) pour les fonctionnalités cloud

---

## 🚀 Installation

### 1. Cloner le dépôt

```bash
git clone https://github.com/YassineCHN/SISE_ULTIMATE_GAMES.git
cd SISE_ULTIMATE_GAMES
```

### 2. Créer un environnement virtuel

Choisissez l'une des trois méthodes selon votre gestionnaire de paquets :

**pip + venv** (standard)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**conda**
```bash
conda create -n sise-games python=3.10
conda activate sise-games
```

**uv** (recommandé — ultra-rapide)
```bash
uv venv --python 3.10
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Installer les dépendances

```bash
# pip / venv
pip install -r requirements.txt

# conda
pip install -r requirements.txt

# uv (recommandé — lit pyproject.toml + uv.lock)
uv sync
```

### 4. Configurer les variables d'environnement

```bash
cp .env.example .env
# puis remplir les valeurs dans .env
```

| Variable | Où la trouver |
|---|---|
| `SUPABASE_URL` | Dashboard Supabase → Settings → API → Project URL |
| `SUPABASE_KEY` | Dashboard Supabase → Settings → API → anon public key |
| `MISTRAL_API_KEY` | [console.mistral.ai/api-keys](https://console.mistral.ai/api-keys) |

> **Sans ces clés**, l'application fonctionne en mode dégradé :
> les sessions sont sauvegardées localement en CSV et les résumés IA sont générés localement (mock).

---

## 🎮 Utilisation

### Workflow recommandé : tout depuis le dashboard

L'application est conçue pour être pilotée depuis le **dashboard Dash**. C'est lui qui lance les jeux, affiche les stats en temps réel et centralise toutes les données.

**1. Démarrer le dashboard**

```bash
cd app
python app.py
# ou avec uv
uv run app.py
```

Ouvrir [http://localhost:8050](http://localhost:8050) dans votre navigateur.

**2. Jouer depuis la page "Live Session"**

- Saisir votre nom de joueur
- Choisir un jeu (Reflex, Labyrinthe, Shooter, Racing)
- Cliquer sur **Lancer** — la fenêtre Pygame s'ouvre, les inputs sont capturés en temps réel
- Le dashboard se met à jour en direct : flux d'inputs, score, métriques manette
- À la fin de la session, le résumé IA (Mistral) apparaît automatiquement dans l'onglet **Résumés**

**3. Explorer vos données**

| Page dashboard | Contenu |
|---|---|
| 🎮 **Live Session** | Lancement des jeux + visualisation temps réel des inputs |
| 👤 **Profils** | Clustering UMAP, profil attribué, comparaison entre joueurs |
| 🏆 **Leaderboard** | Classements par jeu et global |
| 📋 **Résumés IA** | Analyses Mistral post-session avec conseils personnalisés |
| 💬 **Chat IA** | Coach conversationnel avec accès à l'historique des sessions |
| 🤖 **Agent IA** | Lancer l'agent imitateur depuis l'interface |

### Lancement direct en ligne de commande (optionnel)

Il est aussi possible de lancer un jeu sans passer par le dashboard :

```bash
python main.py <jeu> <nom_joueur>
# ou avec flags nommés
python main.py --game <jeu> --player <nom_joueur>
# Exemple
python main.py shooter Modou
# ou avec uv
uv run main.py shooter Modou
```

> ⚠️ Dans ce cas, les stats ne sont pas visualisées en temps réel — uniquement sauvegardées dans Supabase et consultables ensuite dans le dashboard.

### Analyse shooter standalone

`analysis_shooter.py` peut être lancé indépendamment du dashboard pour générer des graphiques matplotlib (clustering, progression, corrélations) depuis les données Supabase ou le CSV local :

```bash
python analysis_shooter.py
# ou avec uv
uv run analysis_shooter.py
# Les graphiques sont sauvegardés dans outputs/
```

### Diagnostiquer la manette

`test_controller.py` est un outil de diagnostic graphique (Pygame) qui affiche en temps réel tous les axes bruts, boutons et le mapping interprété par le module `Controller` :

```bash
python main.py --test
# ou directement
python test_controller.py
# ou avec uv
uv run test_controller.py
```

**Manettes testées et supportées :**

| Manette | Détection automatique |
|---|---|
| Xbox 360 / Xbox One / Series | Oui (`"xbox"`, `"xinput"` dans le nom) |
| PS4 DualShock 4 | Oui (`"dualshock"`, `"wireless controller"`, `"sony"`) |
| PS5 DualSense | Oui (`"dualsense"`) |
| PS3 DualShock 3 | Oui (`"ps3"`, `"sixaxis"`) — nécessite SCP Toolkit sur Windows |
| Manette générique USB | Fallback automatique (mapping PS) |
| Clavier | Fallback si aucune manette détectée |

> **Aucune manette disponible ?** Le fallback clavier est activé automatiquement :
> `Flèches` = joystick gauche · `Z/X/C/V` = boutons · `A/E` = gâchettes

### Adapter le mapping à une manette non reconnue

Si votre manette fonctionne partiellement (axes inversés, mauvais boutons), lancez d'abord `test_controller.py` pour identifier les indices d'axes bruts, puis modifiez le profil `"generic"` dans `core/controller.py` :

```python
# core/controller.py — lignes 38-50
AXIS_MAP = {
    ...
    "generic": {"lx": 0, "ly": 1, "rx": 2, "ry": 3, "lt": 4, "rt": 5},
    #            ↑ remplacer les indices selon ce qu'affiche test_controller.py
}
L1R1_MAP = {
    ...
    "generic": (4, 5),   # indices des boutons L1, R1
}
```

---

## 🎮 Tableau des commandes

### Reflex

| Action | Clavier | Manette |
|---|---|---|
| Bouton A / Croix | `Z` | Bouton A / Croix |
| Bouton B / Rond | `X` | Bouton B / Rond |
| Bouton X / Carré | `C` | Bouton X / Carré |
| Bouton Y / Triangle | `V` | Bouton Y / Triangle |

### Labyrinthe

| Action | Clavier | Manette |
|---|---|---|
| Se déplacer | Flèches directionnelles | Joystick gauche |

### Shooter

| Action | Clavier | Manette |
|---|---|---|
| Se déplacer | `W A S D` | Joystick gauche |
| Viser | Flèches directionnelles | Joystick droit |
| Tirer | `ESPACE` | Gâchette droite (RT/R2) |
| Dash / Boost | `SHIFT` | Gâchette gauche (LT/L2) |
| Bombe | `K` | Bouton Y / Triangle |

### Racing

| Action | Clavier | Manette |
|---|---|---|
| Accélérer | `↑` | Gâchette droite (RT/R2) |
| Freiner | `↓` | Gâchette gauche (LT/L2) |
| Tourner | `← →` | Joystick gauche (axe X) |
| Nitro | `SHIFT` | Bouton A / Croix |

> **Sans manette**, le fallback clavier est activé automatiquement au lancement du jeu.

---

## 📊 Features extraites par session

Chaque session génère **20 features** utilisées pour le clustering :

| Catégorie | Features |
|---|---|
| **Boutons** | fréquence d'appuis, variété, durée moyenne de maintien |
| **Joystick gauche** | position moyenne X/Y, variabilité, changements de direction |
| **Joystick droit** | position moyenne X/Y, variabilité |
| **Gâchettes** | utilisation moyenne L/R, brutalité (brusquerie des inputs) |
| **Timing** | temps de réaction moyen, régularité des inputs |
| **Score** | score final de la session |

---

## 🧩 Profils de joueurs

Le clustering UMAP + K-Means identifie 4 archétypes :

| Profil | Caractéristiques |
|---|---|
| 🔴 **Agressif** | Haute fréquence de boutons, mouvements amples, gâchettes brutales |
| 🔵 **Prudent** | Peu d'inputs, mouvements lents et calculés, gâchettes douces |
| 🟢 **Précis** | Joystick stable, variabilité faible, régularité élevée |
| 🟡 **Chaotique** | Inputs imprévisibles, timing irrégulier, forte variabilité |

---

## 🛠️ Stack technique

| Composant | Technologie |
|---|---|
| Mini-jeux | [Pygame](https://www.pygame.org/) 2.6 |
| Dashboard | [Dash](https://dash.plotly.com/) 2.17 + [Plotly](https://plotly.com/) 5.22 |
| ML | [scikit-learn](https://scikit-learn.org/) 1.4 + [UMAP](https://umap-learn.readthedocs.io/) 0.5 |
| LLM / Coach IA | [Mistral AI](https://mistral.ai/) (mistral-medium) |
| Base de données | [Supabase](https://supabase.com/) (PostgreSQL) |
| Data | [NumPy](https://numpy.org/) 1.26 + [Pandas](https://pandas.pydata.org/) 2.2 |

---

## 👥 Équipe

Projet réalisé dans le cadre du **Challenge IA Temps Réel — Master SISE 2025–2026**, Université Lyon 2.

| Membre | GitHub |
|---|---|
| **Modou Mboup** | [@Modou010](https://github.com/Modou010) |
| **Yassine Cheniour** | [@YassineCHN](https://github.com/YassineCHN) |
| **Nico Dena** | [@DenaNico1](https://github.com/DenaNico1) |

🔗 **Dépôt GitHub :** [YassineCHN/SISE_ULTIMATE_GAMES](https://github.com/YassineCHN/SISE_ULTIMATE_GAMES/tree/main)

---

## 🗄️ Schéma Supabase (nouveau projet)

> **Uniquement si vous utilisez votre propre instance Supabase.** Les `.env` fournis par l'équipe pointent vers une instance déjà configurée.

```sql
create table sessions (
  id bigint generated always as identity primary key,
  created_at timestamptz default now(),
  player_name text, game_id text, duration_sec float,
  btn_press_rate float, btn_variety float, btn_hold_avg_ms float,
  lx_mean float, ly_mean float, lx_std float, ly_std float, lx_direction_changes float,
  rx_mean float, ry_mean float, rx_std float, ry_std float,
  lt_mean float, rt_mean float, lt_brutality float, rt_brutality float,
  reaction_time_avg_ms float, input_regularity float, score int,
  source text default 'unknown'
);

create table summaries (
  id bigint generated always as identity primary key,
  created_at timestamptz default now(),
  player_name text, game_id text, summary_md text
);

create table inputs_live (
  id bigint generated always as identity primary key,
  captured_at timestamptz default now(),
  player_name text, game_id text, session_token text,
  lx float, ly float, rx float, ry float, lt float, rt float,
  btn_a bool, btn_b bool, btn_x bool, btn_y bool, event_type text
);

create table profils_ml (
  player_name text primary key,
  updated_at timestamptz default now(),
  cluster_id int, cluster_name text, features_json jsonb
);
```

---

## 📄 Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
