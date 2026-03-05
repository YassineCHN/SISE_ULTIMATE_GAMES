# 🎮 SISE Ultimate Games — Controller Profiler

> **Projet IA Temps Réel · Master SISE 2025–2026**  
> Analyse comportementale de joueurs via manette Xbox/PS en temps réel, clustering de profils et coaching IA.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Dash](https://img.shields.io/badge/Dashboard-Dash%2FPlotly-informational?logo=plotly)](https://dash.plotly.com/)
[![Supabase](https://img.shields.io/badge/Backend-Supabase-3ECF8E?logo=supabase)](https://supabase.com/)
[![Mistral](https://img.shields.io/badge/LLM-Mistral%20AI-orange)](https://mistral.ai/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

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
| 🧠 **Clustering UMAP + K-Means** | 4 profils de joueurs : Agressif, Prudent, Précis, Chaotique |
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
│   ├── requirements.txt     # Dépendances spécifiques au dashboard
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
conda install --file requirements.txt

# uv
uv pip install -r requirements.txt
```

### 4. Configurer les variables d'environnement

Créer un fichier `.env` à la racine du projet :

```env
SUPABASE_URL=https://votre-projet.supabase.co
SUPABASE_KEY=votre_clé_anon_publique
MISTRAL_API_KEY=votre_clé_api_mistral
```

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
python main.py <nom_joueur> <jeu>
# Exemple
python main.py Modou shooter
```

> ⚠️ Dans ce cas, les stats ne sont pas visualisées en temps réel — uniquement sauvegardées dans Supabase et consultables ensuite dans le dashboard.

### Diagnostiquer la manette

```bash
python test_controller.py
```

---

## 📊 Features extraites par session

Chaque session génère **21 features** utilisées pour le clustering :

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

## 📄 Licence

Ce projet est distribué sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.
