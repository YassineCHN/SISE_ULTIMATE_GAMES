import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import sys
import os
import subprocess

# Chemin racine du projet (parent de app/)
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GAMES = {
    "reflex": "games/reflex_game.py",
    "labyrinth": "games/labyrinth_game.py",
    "shooter": "games/shooter_game.py",
    "racing": "games/racing_game.py",
}

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT_DIR, ".env"))
_SUPABASE_URL = os.getenv("SUPABASE_URL")
_SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_AVAILABLE = bool(_SUPABASE_URL and _SUPABASE_KEY)
if SUPABASE_AVAILABLE:
    try:
        from core.supabase_client import fetch_all_sessions, fetch_latest_sessions
    except Exception:
        SUPABASE_AVAILABLE = False

app = dash.Dash(
    __name__,
    assets_folder="assets",
    suppress_callback_exceptions=True,
    title="SISE Gaming — Controller Profiler",
)

# ─────────────────────────────────────────────
# MOCK DATA
# ─────────────────────────────────────────────
np.random.seed(42)
N = 60
cluster_centers = {0: (2.0, 1.5), 1: (-2.0, 1.0), 2: (0.5, -2.5), 3: (-0.5, 2.5)}
cluster_names = {0: "Agressif", 1: "Prudent", 2: "Précis", 3: "Chaotique"}
cluster_colors_map = {0: "#FF4C6A", 1: "#00E5FF", 2: "#69FF47", 3: "#FFB800"}

mock_players = [
    "Thomas",
    "Emma",
    "Lucas",
    "Léa",
    "Noah",
    "Chloé",
    "Ethan",
    "Inès",
    "Hugo",
    "Camille",
    "Théo",
    "Jade",
    "Louis",
    "Manon",
    "Nathan",
    "Alice",
    "Axel",
    "Lucie",
    "Maxime",
    "Sarah",
    "Raphaël",
]
umap_x, umap_y, labels, player_names = [], [], [], []
for i in range(N):
    c = i % 4
    cx, cy = cluster_centers[c]
    umap_x.append(cx + np.random.randn() * 0.6)
    umap_y.append(cy + np.random.randn() * 0.6)
    labels.append(c)
    player_names.append(mock_players[i % len(mock_players)])

df_umap_mock = pd.DataFrame(
    {
        "x": umap_x,
        "y": umap_y,
        "cluster": [cluster_names[l] for l in labels],
        "player": player_names,
    }
)

features_list = [
    "Réactivité",
    "Agressivité",
    "Fluidité",
    "Précision",
    "Prise de risque",
    "Consistance",
]
radar_profiles = {
    "Agressif": [0.4, 0.95, 0.5, 0.6, 0.9, 0.5],
    "Prudent": [0.8, 0.2, 0.75, 0.85, 0.2, 0.9],
    "Précis": [0.9, 0.5, 0.9, 0.95, 0.5, 0.85],
    "Chaotique": [0.6, 0.8, 0.3, 0.4, 0.85, 0.3],
}


# ─────────────────────────────────────────────
# DATA LOADER
# ─────────────────────────────────────────────
def load_real_df():
    if not SUPABASE_AVAILABLE:
        return None
    try:
        data = fetch_all_sessions()
        if data and len(data) >= 3:
            return pd.DataFrame(data)
    except Exception as e:
        print(f"⚠️ Fallback mock : {e}")
    return None


def build_umap_df(df_real):
    if df_real is not None:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA

            num_cols = [
                "btn_press_rate",
                "lx_std",
                "ly_std",
                "reaction_time_avg_ms",
                "input_regularity",
                "score",
            ]
            available = [c for c in num_cols if c in df_real.columns]
            X = df_real[available].fillna(0).values
            if X.shape[0] >= 3 and X.shape[1] >= 2:
                coords = PCA(n_components=2).fit_transform(
                    StandardScaler().fit_transform(X)
                )
                return pd.DataFrame(
                    {
                        "x": coords[:, 0],
                        "y": coords[:, 1],
                        "cluster": df_real["game_id"].values,
                        "player": df_real["player_name"].values,
                    }
                )
        except Exception as e:
            print(f"⚠️ PCA échoué : {e}")
    return df_umap_mock


# ─────────────────────────────────────────────
# THEMES
# ─────────────────────────────────────────────
THEMES = {
    "cyberpunk": {
        "name": "⚡ Cyberpunk",
        "bg": "#0A0A0F",
        "sidebar": "#0D0D1A",
        "card": "#12121F",
        "border": "#7B2FBE",
        "accent1": "#C724B1",
        "accent2": "#00F5FF",
        "accent3": "#FF4C6A",
        "text": "#E8E8FF",
        "subtext": "#8888AA",
        "font": "'Orbitron', monospace",
        "font_body": "'Share Tech Mono', monospace",
        "glow": "0 0 20px rgba(199,36,177,0.4)",
        "gradient": "linear-gradient(135deg, #C724B1 0%, #00F5FF 100%)",
    },
    "scientific": {
        "name": "🔬 Scientific",
        "bg": "#0B1120",
        "sidebar": "#0E1628",
        "card": "#111C35",
        "border": "#1E3A5F",
        "accent1": "#2979FF",
        "accent2": "#00BCD4",
        "accent3": "#FF6B35",
        "text": "#E3EAF4",
        "subtext": "#7A90B0",
        "font": "'Exo 2', sans-serif",
        "font_body": "'IBM Plex Mono', monospace",
        "glow": "0 0 20px rgba(41,121,255,0.3)",
        "gradient": "linear-gradient(135deg, #2979FF 0%, #00BCD4 100%)",
    },
    "matrix": {
        "name": "🟢 Matrix",
        "bg": "#030A03",
        "sidebar": "#050F05",
        "card": "#071207",
        "border": "#0D3B0D",
        "accent1": "#00FF41",
        "accent2": "#39FF14",
        "accent3": "#ADFF2F",
        "text": "#C8FFC8",
        "subtext": "#4A8A4A",
        "font": "'VT323', monospace",
        "font_body": "'Courier Prime', monospace",
        "glow": "0 0 20px rgba(0,255,65,0.4)",
        "gradient": "linear-gradient(135deg, #00FF41 0%, #39FF14 100%)",
    },
    "datasci": {
        "name": "📊 DataSci",
        "bg": "#0F0E17",
        "sidebar": "#13121F",
        "card": "#1A1929",
        "border": "#2D2B45",
        "accent1": "#FF6B35",
        "accent2": "#F7C59F",
        "accent3": "#FFFFFE",
        "text": "#FFFFFE",
        "subtext": "#A7A9BE",
        "font": "'Syne', sans-serif",
        "font_body": "'Space Mono', monospace",
        "glow": "0 0 20px rgba(255,107,53,0.35)",
        "gradient": "linear-gradient(135deg, #FF6B35 0%, #F7C59F 100%)",
    },
}


# ─────────────────────────────────────────────
# HELPERS UI
# ─────────────────────────────────────────────
def make_card(children, theme, style_extra=None):
    t = THEMES[theme]
    style = {
        "background": t["card"],
        "border": f"1px solid {t['border']}",
        "borderRadius": "12px",
        "padding": "20px",
        "boxShadow": t["glow"],
    }
    if style_extra:
        style.update(style_extra)
    return html.Div(children, style=style)


def stat_card(label, value, delta, theme):
    t = THEMES[theme]
    return make_card(
        [
            html.Div(
                label,
                style={
                    "color": t["subtext"],
                    "fontSize": "11px",
                    "textTransform": "uppercase",
                    "letterSpacing": "2px",
                    "marginBottom": "8px",
                },
            ),
            html.Div(
                value,
                style={
                    "color": t["accent1"],
                    "fontSize": "28px",
                    "fontWeight": "700",
                    "fontFamily": t["font"],
                },
            ),
            html.Div(
                delta,
                style={"color": t["accent2"], "fontSize": "12px", "marginTop": "4px"},
            ),
        ],
        theme,
        {"flex": "1", "minWidth": "140px"},
    )


def data_badge(is_real, theme):
    t = THEMES[theme]
    if is_real:
        return html.Div(
            "🟢 LIVE — Supabase",
            style={
                "color": t["accent2"],
                "fontSize": "10px",
                "letterSpacing": "2px",
                "border": f"1px solid {t['accent2']}",
                "borderRadius": "4px",
                "padding": "2px 8px",
                "display": "inline-block",
            },
        )
    return html.Div(
        "🟡 MOCK — En attente de sessions",
        style={
            "color": t["accent3"],
            "fontSize": "10px",
            "letterSpacing": "2px",
            "border": f"1px solid {t['accent3']}",
            "borderRadius": "4px",
            "padding": "2px 8px",
            "display": "inline-block",
        },
    )


def make_inputs_table(theme, rows):
    """Tableau temps réel des derniers inputs — remplace le graphique simulé."""
    t = THEMES[theme]

    # Colonnes à afficher
    cols = ["#", "Joueur", "Jeu", "LX", "LY", "LT", "RT", "A", "B", "X", "Y", "Source"]

    header = html.Thead(
        html.Tr(
            [
                html.Th(
                    c,
                    style={
                        "color": t["subtext"],
                        "fontSize": "10px",
                        "padding": "6px 8px",
                        "textTransform": "uppercase",
                        "letterSpacing": "1px",
                        "borderBottom": f"1px solid {t['border']}",
                        "textAlign": "center",
                        "whiteSpace": "nowrap",
                    },
                )
                for c in cols
            ]
        )
    )

    btn_colors = {"A": "#69FF47", "B": "#FF4C6A", "X": "#00E5FF", "Y": "#FFB800"}

    def btn_cell(val, color):
        active = bool(val)
        return html.Td(
            "●" if active else "○",
            style={
                "color": color if active else t["subtext"],
                "fontSize": "14px",
                "textAlign": "center",
                "padding": "5px 6px",
                "textShadow": f"0 0 6px {color}" if active else "none",
            },
        )

    def val_cell(v, accent=False):
        try:
            fv = float(v)
            color = (
                t["accent1"] if accent else (t["accent2"] if fv > 0.3 else t["text"])
            )
        except Exception:
            color = t["text"]
        return html.Td(
            f"{float(v):.2f}" if v not in (None, "", "—") else "—",
            style={
                "color": color,
                "fontSize": "11px",
                "textAlign": "center",
                "padding": "5px 8px",
                "fontFamily": t["font_body"],
            },
        )

    if not rows:
        # Lignes mock pour montrer la structure
        mock_rows = [
            {
                "idx": i + 1,
                "player": "—",
                "game": "—",
                "lx": 0.0,
                "ly": 0.0,
                "lt": 0.0,
                "rt": 0.0,
                "btn_a": False,
                "btn_b": False,
                "btn_x": False,
                "btn_y": False,
                "source": "—",
            }
            for i in range(5)
        ]
    else:
        mock_rows = [
            {
                "idx": len(rows) - i,
                "player": r.get("player_name", "—"),
                "game": r.get("game_id", "—"),
                "lx": r.get("lx", 0),
                "ly": r.get("ly", 0),
                "lt": r.get("lt", 0),
                "rt": r.get("rt", 0),
                "btn_a": r.get("btn_a", False),
                "btn_b": r.get("btn_b", False),
                "btn_x": r.get("btn_x", False),
                "btn_y": r.get("btn_y", False),
                "source": r.get("event_type", "—"),
            }
            for i, r in enumerate(reversed(rows[-10:]))
        ]

    game_colors = {
        "reflex": "#FF4C6A",
        "labyrinth": "#00E5FF",
        "shooter": "#69FF47",
        "racing": "#FFB800",
    }
    source_icons = {"controller": "🎮", "keyboard": "⌨️", "—": "⏳"}

    body_rows = []
    for row in mock_rows:
        gc = game_colors.get(row["game"], t["subtext"])
        body_rows.append(
            html.Tr(
                [
                    html.Td(
                        str(row["idx"]),
                        style={
                            "color": t["subtext"],
                            "fontSize": "10px",
                            "textAlign": "center",
                            "padding": "5px 8px",
                        },
                    ),
                    html.Td(
                        row["player"],
                        style={
                            "color": t["text"],
                            "fontSize": "11px",
                            "padding": "5px 8px",
                            "fontFamily": t["font_body"],
                        },
                    ),
                    html.Td(
                        row["game"],
                        style={
                            "color": gc,
                            "fontSize": "11px",
                            "padding": "5px 8px",
                            "fontFamily": t["font_body"],
                        },
                    ),
                    val_cell(row["lx"]),
                    val_cell(row["ly"]),
                    val_cell(row["lt"], accent=True),
                    val_cell(row["rt"], accent=True),
                    btn_cell(row["btn_a"], btn_colors["A"]),
                    btn_cell(row["btn_b"], btn_colors["B"]),
                    btn_cell(row["btn_x"], btn_colors["X"]),
                    btn_cell(row["btn_y"], btn_colors["Y"]),
                    html.Td(
                        source_icons.get(row["source"], "⏳"),
                        style={
                            "textAlign": "center",
                            "padding": "5px 8px",
                            "fontSize": "13px",
                        },
                    ),
                ],
                style={
                    "borderBottom": f"1px solid {t['border']}",
                    "transition": "background 0.2s",
                },
            )
        )

    return html.Table(
        [header, html.Tbody(body_rows)],
        style={"width": "100%", "borderCollapse": "collapse", "tableLayout": "fixed"},
    )


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────
def make_umap_fig(theme, df_real=None):
    t = THEMES[theme]
    plot_df = build_umap_df(df_real)
    colors_by_group = {
        "reflex": "#FF4C6A",
        "labyrinth": "#00E5FF",
        "shooter": "#69FF47",
        "racing": "#FFB800",
        "Agressif": "#FF4C6A",
        "Prudent": "#00E5FF",
        "Précis": "#69FF47",
        "Chaotique": "#FFB800",
    }
    fig = go.Figure()
    for group in plot_df["cluster"].unique():
        mask = plot_df["cluster"] == group
        color = colors_by_group.get(group, "#AAAAAA")
        fig.add_trace(
            go.Scatter(
                x=plot_df[mask]["x"],
                y=plot_df[mask]["y"],
                mode="markers+text",
                name=str(group),
                text=plot_df[mask]["player"],
                textposition="top center",
                textfont=dict(size=8, color=color),
                marker=dict(
                    size=10,
                    color=color,
                    opacity=0.85,
                    line=dict(width=1, color="white"),
                ),
                hovertemplate="<b>%{text}</b><br>Groupe: "
                + str(group)
                + "<extra></extra>",
            )
        )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=t["border"], borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor=t["border"], zeroline=False, title="Axe 1"),
        yaxis=dict(showgrid=True, gridcolor=t["border"], zeroline=False, title="Axe 2"),
        margin=dict(l=20, r=20, t=20, b=20),
        height=380,
    )
    return fig


def make_radar_fig(profile_name, theme):
    t = THEMES[theme]
    vals = radar_profiles.get(profile_name, radar_profiles["Précis"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=vals + [vals[0]],
            theta=features_list + [features_list[0]],
            fill="toself",
            fillcolor=f"rgba({int(t['accent1'][1:3],16)},{int(t['accent1'][3:5],16)},{int(t['accent1'][5:7],16)},0.25)",
            line=dict(color=t["accent1"], width=2),
            name=profile_name,
        )
    )
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(
                visible=True, range=[0, 1], gridcolor=t["border"], color=t["subtext"]
            ),
            angularaxis=dict(gridcolor=t["border"], color=t["text"]),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        margin=dict(l=30, r=30, t=30, b=30),
        height=300,
        showlegend=False,
    )
    return fig


def make_reaction_hist(theme, df_real=None):
    t = THEMES[theme]
    data = None
    if df_real is not None and "reaction_time_avg_ms" in df_real.columns:
        tmp = df_real["reaction_time_avg_ms"].dropna().values
        if len(tmp[tmp > 0]) > 0:
            data = tmp[tmp > 0]
    if data is None or len(data) == 0:
        data = np.concatenate(
            [
                np.random.normal(180, 20, 30),
                np.random.normal(240, 30, 25),
                np.random.normal(310, 25, 20),
            ]
        )
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=data, nbinsx=25, marker_color=t["accent1"], opacity=0.8)
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        xaxis=dict(showgrid=False, title="Réaction (ms)"),
        yaxis=dict(showgrid=True, gridcolor=t["border"]),
        margin=dict(l=20, r=20, t=10, b=20),
        height=220,
        showlegend=False,
        bargap=0.05,
    )
    return fig


def make_score_bar(theme, df_real=None):
    t = THEMES[theme]
    if df_real is not None and "score" in df_real.columns:
        grp = df_real.groupby("game_id")["score"].mean().reset_index()
        games = grp["game_id"].tolist()
        scores = grp["score"].tolist()
    else:
        games = ["reflex", "labyrinth", "shooter", "racing"]
        scores = [620, 850, 1200, 2400]
    colors = ["#FF4C6A", "#00E5FF", "#69FF47", "#FFB800"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(x=games, y=scores, marker_color=colors[: len(games)], opacity=0.85)
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=t["border"]),
        margin=dict(l=20, r=20, t=10, b=20),
        height=200,
        showlegend=False,
    )
    return fig


def make_agent_comparison(theme):
    t = THEMES[theme]
    cats = ["Réactivité", "Précision", "Fluidité", "Agressivité", "Consistance"]
    human = [0.72, 0.65, 0.80, 0.55, 0.68]
    agent = [0.69, 0.67, 0.77, 0.58, 0.71]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(name="Humain", x=cats, y=human, marker_color=t["accent2"], opacity=0.85)
    )
    fig.add_trace(
        go.Bar(
            name="Agent IA", x=cats, y=agent, marker_color=t["accent1"], opacity=0.85
        )
    )
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=t["border"], range=[0, 1]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=10, b=20),
        height=250,
    )
    return fig


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────
def page_game(theme, df_real=None):
    t = THEMES[theme]
    is_real = df_real is not None

    return html.Div(
        [
            # ── Titre + badge ──
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "🎮 Session Live",
                                style={
                                    "color": t["accent1"],
                                    "fontSize": "22px",
                                    "fontWeight": "700",
                                    "fontFamily": t["font"],
                                    "marginBottom": "4px",
                                },
                            ),
                            html.Div(
                                "Capture des inputs manette en temps réel",
                                style={"color": t["subtext"], "fontSize": "13px"},
                            ),
                        ]
                    ),
                    html.Div(id="data-badge-container"),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "marginBottom": "24px",
                },
            ),
            # ── Métriques ──
            html.Div(
                [
                    html.Div(
                        id="stat-sessions", style={"flex": "1", "minWidth": "140px"}
                    ),
                    html.Div(
                        id="stat-players", style={"flex": "1", "minWidth": "140px"}
                    ),
                    html.Div(id="stat-score", style={"flex": "1", "minWidth": "140px"}),
                    html.Div(
                        id="stat-reaction", style={"flex": "1", "minWidth": "140px"}
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "marginBottom": "24px",
                    "flexWrap": "wrap",
                },
            ),
            # ── Lancer un jeu (remonté ici) ──
            make_card(
                [
                    html.Div(
                        "Lancer un jeu",
                        style={
                            "color": t["subtext"],
                            "fontSize": "11px",
                            "textTransform": "uppercase",
                            "letterSpacing": "2px",
                            "marginBottom": "16px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Nom du joueur",
                                        style={
                                            "color": t["subtext"],
                                            "fontSize": "11px",
                                            "marginBottom": "6px",
                                        },
                                    ),
                                    dcc.Input(
                                        id="input-player-name",
                                        placeholder="ex: Thomas",
                                        debounce=True,
                                        style={
                                            "background": t["bg"],
                                            "border": f"1px solid {t['border']}",
                                            "color": t["text"],
                                            "padding": "8px 12px",
                                            "borderRadius": "6px",
                                            "fontFamily": t["font_body"],
                                            "width": "180px",
                                        },
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Jeu",
                                        style={
                                            "color": t["subtext"],
                                            "fontSize": "11px",
                                            "marginBottom": "6px",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="dropdown-game-select",
                                        options=[
                                            {"label": "🎯 Reflex", "value": "reflex"},
                                            {
                                                "label": "🌀 Labyrinth",
                                                "value": "labyrinth",
                                            },
                                            {"label": "🚀 Shooter", "value": "shooter"},
                                            {"label": "🏎️ Racing", "value": "racing"},
                                        ],
                                        value="reflex",
                                        clearable=False,
                                        style={
                                            "background": t["bg"],
                                            "color": "#000",
                                            "border": f"1px solid {t['border']}",
                                            "borderRadius": "6px",
                                            "width": "200px",
                                            "fontFamily": t["font_body"],
                                        },
                                    ),
                                ]
                            ),
                            html.Button(
                                "▶ LANCER LE JEU",
                                id="btn-launch-game",
                                n_clicks=0,
                                style={
                                    "background": t["gradient"],
                                    "border": "none",
                                    "color": "#000",
                                    "padding": "10px 24px",
                                    "borderRadius": "6px",
                                    "cursor": "pointer",
                                    "fontFamily": t["font"],
                                    "fontSize": "13px",
                                    "fontWeight": "700",
                                    "letterSpacing": "2px",
                                    "alignSelf": "flex-end",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "16px",
                            "alignItems": "flex-end",
                            "flexWrap": "wrap",
                        },
                    ),
                    html.Div(
                        id="launch-feedback",
                        style={"marginTop": "12px", "fontSize": "12px"},
                    ),
                ],
                theme,
                {"marginBottom": "16px"},
            ),
            make_card(
                [
                    html.Div(
                        [
                            html.Div(
                                "Flux inputs temps réel",
                                style={
                                    "color": t["subtext"],
                                    "fontSize": "11px",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "2px",
                                },
                            ),
                            html.Div(
                                id="live-source-badge"
                            ),  # affiche "🎮 Manette" ou "⌨️ Clavier"
                        ],
                        style={
                            "display": "flex",
                            "justifyContent": "space-between",
                            "marginBottom": "12px",
                        },
                    ),
                    # Graphique joysticks
                    dcc.Graph(
                        id="live-joystick-graph",
                        config={"displayModeBar": False},
                        style={"height": "180px"},
                    ),
                    # Jauges gâchettes
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "LT",
                                        style={
                                            "color": t["subtext"],
                                            "fontSize": "11px",
                                            "marginBottom": "4px",
                                        },
                                    ),
                                    html.Div(
                                        id="gauge-lt",
                                        style={
                                            "background": t["border"],
                                            "borderRadius": "4px",
                                            "height": "8px",
                                        },
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "RT",
                                        style={
                                            "color": t["subtext"],
                                            "fontSize": "11px",
                                            "marginBottom": "4px",
                                        },
                                    ),
                                    html.Div(
                                        id="gauge-rt",
                                        style={
                                            "background": t["border"],
                                            "borderRadius": "4px",
                                            "height": "8px",
                                        },
                                    ),
                                ],
                                style={"flex": "1"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "16px",
                            "marginTop": "12px",
                            "marginBottom": "12px",
                        },
                    ),
                    # Boutons
                    html.Div(id="live-buttons-display"),
                ],
                theme,
                {"flex": "1"},
            ),
            # ── Flux inputs + Distribution réactions ──
            html.Div(
                [
                    # Flux inputs temps réel → tableau
                    make_card(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Inputs manette",
                                        style={
                                            "color": t["subtext"],
                                            "fontSize": "11px",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "2px",
                                        },
                                    ),
                                    html.Div(id="live-source-badge"),
                                ],
                                style={
                                    "display": "flex",
                                    "justifyContent": "space-between",
                                    "alignItems": "center",
                                    "marginBottom": "12px",
                                },
                            ),
                            html.Div(
                                id="live-inputs-table",
                                style={
                                    "overflowX": "auto",
                                    "overflowY": "auto",
                                    "maxHeight": "280px",
                                },
                            ),
                        ],
                        theme,
                        {"flex": "1"},
                    ),
                    # Distribution réactions
                    html.Div(
                        [
                            make_card(
                                [
                                    html.Div(
                                        "Distribution réactions",
                                        style={
                                            "color": t["subtext"],
                                            "fontSize": "11px",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "2px",
                                            "marginBottom": "12px",
                                        },
                                    ),
                                    dcc.Graph(
                                        figure=make_reaction_hist(theme, df_real),
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                theme,
                            ),
                        ],
                        style={
                            "flex": "1",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "flexWrap": "wrap",
                    "marginBottom": "16px",
                },
            ),
            # ── Score moyen par jeu ──
            make_card(
                [
                    html.Div(
                        "Score moyen par jeu",
                        style={
                            "color": t["subtext"],
                            "fontSize": "11px",
                            "textTransform": "uppercase",
                            "letterSpacing": "2px",
                            "marginBottom": "12px",
                        },
                    ),
                    dcc.Graph(
                        figure=make_score_bar(theme, df_real),
                        config={"displayModeBar": False},
                    ),
                ],
                theme,
            ),
        ]
    )


def page_profils(theme, df_real=None):
    t = THEMES[theme]
    is_real = df_real is not None
    n_players = str(df_real["player_name"].nunique()) if is_real else "21 (mock)"
    n_sess = str(len(df_real)) if is_real else "60 (mock)"
    axe_label = "PCA 2D (UMAP après ML)" if is_real else "UMAP 2D (mock)"

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "🧬 Profils comportementaux",
                                style={
                                    "color": t["accent1"],
                                    "fontSize": "22px",
                                    "fontWeight": "700",
                                    "fontFamily": t["font"],
                                    "marginBottom": "4px",
                                },
                            ),
                            html.Div(
                                "Clustering — projection des joueurs par style de jeu",
                                style={"color": t["subtext"], "fontSize": "13px"},
                            ),
                        ]
                    ),
                    data_badge(is_real, theme),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "marginBottom": "24px",
                },
            ),
            html.Div(
                [
                    stat_card("Joueurs", n_players, "Analysés", theme),
                    stat_card("Sessions", n_sess, "Au total", theme),
                    stat_card("Clusters", "4", "K-Means optimal", theme),
                    stat_card("Features", "7", "Comportementales", theme),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "marginBottom": "24px",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(
                [
                    make_card(
                        [
                            html.Div(
                                axe_label,
                                style={
                                    "color": t["subtext"],
                                    "fontSize": "11px",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "2px",
                                    "marginBottom": "12px",
                                },
                            ),
                            dcc.Graph(
                                id="umap-graph",
                                figure=make_umap_fig(theme, df_real),
                                config={"displayModeBar": False},
                            ),
                        ],
                        theme,
                        {"flex": "2"},
                    ),
                    make_card(
                        [
                            html.Div(
                                "Profil sélectionné",
                                style={
                                    "color": t["subtext"],
                                    "fontSize": "11px",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "2px",
                                    "marginBottom": "12px",
                                },
                            ),
                            dcc.Dropdown(
                                id="profile-selector",
                                options=[
                                    {"label": p, "value": p}
                                    for p in [
                                        "Agressif",
                                        "Prudent",
                                        "Précis",
                                        "Chaotique",
                                    ]
                                ],
                                value="Précis",
                                style={
                                    "background": t["card"],
                                    "color": t["text"],
                                    "border": f"1px solid {t['border']}",
                                    "borderRadius": "6px",
                                    "marginBottom": "12px",
                                },
                                className="custom-dropdown",
                            ),
                            dcc.Graph(
                                id="radar-graph",
                                figure=make_radar_fig("Précis", theme),
                                config={"displayModeBar": False},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Span(
                                                "● Agressif  ",
                                                style={
                                                    "color": "#FF4C6A",
                                                    "fontFamily": t["font_body"],
                                                    "fontSize": "12px",
                                                },
                                            ),
                                            html.Span(
                                                "● Prudent  ",
                                                style={
                                                    "color": "#00E5FF",
                                                    "fontFamily": t["font_body"],
                                                    "fontSize": "12px",
                                                },
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "● Précis  ",
                                                style={
                                                    "color": "#69FF47",
                                                    "fontFamily": t["font_body"],
                                                    "fontSize": "12px",
                                                },
                                            ),
                                            html.Span(
                                                "● Chaotique",
                                                style={
                                                    "color": "#FFB800",
                                                    "fontFamily": t["font_body"],
                                                    "fontSize": "12px",
                                                },
                                            ),
                                        ]
                                    ),
                                ],
                                style={"marginTop": "8px"},
                            ),
                        ],
                        theme,
                        {"flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
            html.Div(
                [
                    make_card(
                        [
                            html.Div(
                                "Features moyennes par cluster",
                                style={
                                    "color": t["subtext"],
                                    "fontSize": "11px",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "2px",
                                    "marginBottom": "16px",
                                },
                            ),
                            html.Table(
                                [
                                    html.Thead(
                                        html.Tr(
                                            [
                                                html.Th(
                                                    col,
                                                    style={
                                                        "color": t["subtext"],
                                                        "fontSize": "11px",
                                                        "padding": "8px 12px",
                                                        "textAlign": "left",
                                                        "borderBottom": f"1px solid {t['border']}",
                                                    },
                                                )
                                                for col in [
                                                    "Profil",
                                                    "Réactivité",
                                                    "Agressivité",
                                                    "Fluidité",
                                                    "Précision",
                                                    "Risque",
                                                    "Consistance",
                                                    "Nb joueurs",
                                                ]
                                            ]
                                        )
                                    ),
                                    html.Tbody(
                                        [
                                            html.Tr(
                                                [
                                                    html.Td(
                                                        v,
                                                        style={
                                                            "color": (
                                                                t["text"]
                                                                if i > 0
                                                                else color
                                                            ),
                                                            "padding": "8px 12px",
                                                            "fontFamily": t[
                                                                "font_body"
                                                            ],
                                                            "fontSize": "13px",
                                                            "borderBottom": f"1px solid {t['border']}",
                                                        },
                                                    )
                                                    for i, v in enumerate(row)
                                                ]
                                            )
                                            for row, color in [
                                                (
                                                    [
                                                        "Agressif",
                                                        "0.40",
                                                        "0.95",
                                                        "0.50",
                                                        "0.60",
                                                        "0.90",
                                                        "0.50",
                                                        "—",
                                                    ],
                                                    "#FF4C6A",
                                                ),
                                                (
                                                    [
                                                        "Prudent",
                                                        "0.80",
                                                        "0.20",
                                                        "0.75",
                                                        "0.85",
                                                        "0.20",
                                                        "0.90",
                                                        "—",
                                                    ],
                                                    "#00E5FF",
                                                ),
                                                (
                                                    [
                                                        "Précis",
                                                        "0.90",
                                                        "0.50",
                                                        "0.90",
                                                        "0.95",
                                                        "0.50",
                                                        "0.85",
                                                        "—",
                                                    ],
                                                    "#69FF47",
                                                ),
                                                (
                                                    [
                                                        "Chaotique",
                                                        "0.60",
                                                        "0.80",
                                                        "0.30",
                                                        "0.40",
                                                        "0.85",
                                                        "0.30",
                                                        "—",
                                                    ],
                                                    "#FFB800",
                                                ),
                                            ]
                                        ]
                                    ),
                                ],
                                style={"width": "100%", "borderCollapse": "collapse"},
                            ),
                        ],
                        theme,
                    ),
                ],
                style={"marginTop": "16px"},
            ),
        ]
    )


def page_classifier(theme, df_real=None):
    t = THEMES[theme]
    is_real = df_real is not None

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "🎯 Classificateur",
                                style={
                                    "color": t["accent1"],
                                    "fontSize": "22px",
                                    "fontWeight": "700",
                                    "fontFamily": t["font"],
                                    "marginBottom": "4px",
                                },
                            ),
                            html.Div(
                                "Identification du profil d'un nouveau joueur en temps réel",
                                style={"color": t["subtext"], "fontSize": "13px"},
                            ),
                        ]
                    ),
                    data_badge(is_real, theme),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "marginBottom": "24px",
                },
            ),
            html.Div(
                [
                    make_card(
                        [
                            html.Div(
                                "Nouveau joueur",
                                style={
                                    "color": t["subtext"],
                                    "fontSize": "11px",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "2px",
                                    "marginBottom": "16px",
                                },
                            ),
                            html.Div(
                                "Nom",
                                style={
                                    "color": t["subtext"],
                                    "fontSize": "11px",
                                    "marginBottom": "6px",
                                },
                            ),
                            dcc.Input(
                                placeholder="ex: Nouveau joueur",
                                style={
                                    "background": t["bg"],
                                    "border": f"1px solid {t['border']}",
                                    "color": t["text"],
                                    "padding": "8px 12px",
                                    "borderRadius": "6px",
                                    "fontFamily": t["font_body"],
                                    "width": "100%",
                                    "marginBottom": "16px",
                                },
                            ),
                            *[
                                html.Div(
                                    [
                                        html.Div(
                                            feat,
                                            style={
                                                "color": t["subtext"],
                                                "fontSize": "11px",
                                                "marginBottom": "4px",
                                            },
                                        ),
                                        dcc.Slider(
                                            0,
                                            1,
                                            0.01,
                                            value=round(np.random.uniform(0.3, 0.9), 2),
                                            marks=None,
                                            tooltip={"placement": "bottom"},
                                            className="custom-slider",
                                        ),
                                        html.Div(style={"marginBottom": "12px"}),
                                    ]
                                )
                                for feat in features_list
                            ],
                            html.Button(
                                "🔍 CLASSIFIER",
                                style={
                                    "background": t["gradient"],
                                    "border": "none",
                                    "color": "#000",
                                    "padding": "12px 32px",
                                    "borderRadius": "6px",
                                    "cursor": "pointer",
                                    "fontFamily": t["font"],
                                    "fontSize": "14px",
                                    "fontWeight": "700",
                                    "letterSpacing": "2px",
                                    "width": "100%",
                                    "marginTop": "8px",
                                },
                            ),
                        ],
                        theme,
                        {"flex": "1"},
                    ),
                    html.Div(
                        [
                            make_card(
                                [
                                    html.Div(
                                        "Résultat",
                                        style={
                                            "color": t["subtext"],
                                            "fontSize": "11px",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "2px",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                "PROFIL IDENTIFIÉ",
                                                style={
                                                    "color": t["subtext"],
                                                    "fontSize": "11px",
                                                    "letterSpacing": "2px",
                                                },
                                            ),
                                            html.Div(
                                                "🎯 PRÉCIS",
                                                style={
                                                    "color": t["accent1"],
                                                    "fontSize": "36px",
                                                    "fontFamily": t["font"],
                                                    "fontWeight": "700",
                                                    "marginTop": "8px",
                                                },
                                            ),
                                            html.Div(
                                                "Confiance : 87%",
                                                style={
                                                    "color": t["accent2"],
                                                    "fontSize": "14px",
                                                    "marginTop": "4px",
                                                },
                                            ),
                                        ],
                                        style={
                                            "textAlign": "center",
                                            "padding": "20px 0",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                p,
                                                                style={
                                                                    "color": c,
                                                                    "fontFamily": t[
                                                                        "font_body"
                                                                    ],
                                                                    "fontSize": "12px",
                                                                },
                                                            ),
                                                            html.Span(
                                                                f"{v}%",
                                                                style={
                                                                    "color": t[
                                                                        "subtext"
                                                                    ],
                                                                    "fontSize": "12px",
                                                                },
                                                            ),
                                                        ],
                                                        style={
                                                            "display": "flex",
                                                            "justifyContent": "space-between",
                                                            "marginBottom": "4px",
                                                        },
                                                    ),
                                                    html.Div(
                                                        html.Div(
                                                            style={
                                                                "width": f"{v}%",
                                                                "height": "6px",
                                                                "background": c,
                                                                "borderRadius": "3px",
                                                                "transition": "width 0.5s ease",
                                                            }
                                                        ),
                                                        style={
                                                            "background": t["border"],
                                                            "borderRadius": "3px",
                                                            "marginBottom": "10px",
                                                        },
                                                    ),
                                                ]
                                            )
                                            for p, v, c in [
                                                ("Précis", 87, "#69FF47"),
                                                ("Prudent", 8, "#00E5FF"),
                                                ("Agressif", 3, "#FF4C6A"),
                                                ("Chaotique", 2, "#FFB800"),
                                            ]
                                        ]
                                    ),
                                ],
                                theme,
                                {"marginBottom": "16px"},
                            ),
                            make_card(
                                [
                                    html.Div(
                                        "Radar comparatif",
                                        style={
                                            "color": t["subtext"],
                                            "fontSize": "11px",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "2px",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    dcc.Graph(
                                        figure=make_radar_fig("Précis", theme),
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                theme,
                            ),
                        ],
                        style={
                            "flex": "1",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
        ]
    )


def page_agent(theme, df_real=None):
    t = THEMES[theme]
    is_real = df_real is not None
    players = df_real["player_name"].unique().tolist() if is_real else mock_players

    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "🤖 Agent Imitateur",
                                style={
                                    "color": t["accent1"],
                                    "fontSize": "22px",
                                    "fontWeight": "700",
                                    "fontFamily": t["font"],
                                    "marginBottom": "4px",
                                },
                            ),
                            html.Div(
                                "L'IA rejoue les patterns comportementaux d'un profil appris",
                                style={"color": t["subtext"], "fontSize": "13px"},
                            ),
                        ]
                    ),
                    data_badge(is_real, theme),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "flex-start",
                    "marginBottom": "24px",
                },
            ),
            html.Div(
                [
                    make_card(
                        [
                            html.Div(
                                "Configuration de l'agent",
                                style={
                                    "color": t["subtext"],
                                    "fontSize": "11px",
                                    "textTransform": "uppercase",
                                    "letterSpacing": "2px",
                                    "marginBottom": "16px",
                                },
                            ),
                            html.Div(
                                "Profil à imiter",
                                style={
                                    "color": t["subtext"],
                                    "fontSize": "11px",
                                    "marginBottom": "6px",
                                },
                            ),
                            dcc.Dropdown(
                                options=[
                                    {"label": p, "value": p}
                                    for p in [
                                        "Agressif",
                                        "Prudent",
                                        "Précis",
                                        "Chaotique",
                                    ]
                                ],
                                value="Agressif",
                                style={
                                    "background": t["card"],
                                    "color": "#000",
                                    "borderRadius": "6px",
                                    "marginBottom": "16px",
                                },
                            ),
                            html.Div(
                                "Joueur cible",
                                style={
                                    "color": t["subtext"],
                                    "fontSize": "11px",
                                    "marginBottom": "6px",
                                },
                            ),
                            dcc.Dropdown(
                                options=[{"label": p, "value": p} for p in players],
                                placeholder="Sélectionner un joueur...",
                                style={
                                    "background": t["card"],
                                    "color": "#000",
                                    "borderRadius": "6px",
                                    "marginBottom": "16px",
                                },
                            ),
                            html.Div(
                                "Fidélité d'imitation",
                                style={
                                    "color": t["subtext"],
                                    "fontSize": "11px",
                                    "marginBottom": "6px",
                                },
                            ),
                            dcc.Slider(
                                0,
                                100,
                                1,
                                value=80,
                                marks={0: "Libre", 50: "Mixte", 100: "Fidèle"},
                                tooltip={"placement": "bottom"},
                                className="custom-slider",
                            ),
                            html.Div(style={"height": "20px"}),
                            html.Button(
                                "▶ LANCER L'AGENT",
                                style={
                                    "background": t["gradient"],
                                    "border": "none",
                                    "color": "#000",
                                    "padding": "12px 32px",
                                    "borderRadius": "6px",
                                    "cursor": "pointer",
                                    "fontFamily": t["font"],
                                    "fontSize": "14px",
                                    "fontWeight": "700",
                                    "letterSpacing": "2px",
                                    "width": "100%",
                                },
                            ),
                            html.Div(style={"height": "8px"}),
                            html.Button(
                                "⏹ ARRÊTER",
                                style={
                                    "background": "transparent",
                                    "border": f"1px solid {t['accent3']}",
                                    "color": t["accent3"],
                                    "padding": "10px 32px",
                                    "borderRadius": "6px",
                                    "cursor": "pointer",
                                    "fontFamily": t["font"],
                                    "fontSize": "13px",
                                    "width": "100%",
                                },
                            ),
                        ],
                        theme,
                        {"flex": "1"},
                    ),
                    html.Div(
                        [
                            make_card(
                                [
                                    html.Div(
                                        "Comparaison Humain vs Agent IA",
                                        style={
                                            "color": t["subtext"],
                                            "fontSize": "11px",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "2px",
                                            "marginBottom": "12px",
                                        },
                                    ),
                                    dcc.Graph(
                                        figure=make_agent_comparison(theme),
                                        config={"displayModeBar": False},
                                    ),
                                ],
                                theme,
                                {"marginBottom": "16px"},
                            ),
                            make_card(
                                [
                                    html.Div(
                                        "Score de similarité comportementale",
                                        style={
                                            "color": t["subtext"],
                                            "fontSize": "11px",
                                            "textTransform": "uppercase",
                                            "letterSpacing": "2px",
                                            "marginBottom": "16px",
                                        },
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                "92.4%",
                                                style={
                                                    "color": t["accent1"],
                                                    "fontSize": "48px",
                                                    "fontFamily": t["font"],
                                                    "fontWeight": "700",
                                                    "textAlign": "center",
                                                },
                                            ),
                                            html.Div(
                                                "de similarité avec le profil Agressif",
                                                style={
                                                    "color": t["subtext"],
                                                    "fontSize": "13px",
                                                    "textAlign": "center",
                                                    "marginTop": "4px",
                                                },
                                            ),
                                            html.Div(
                                                html.Div(
                                                    style={
                                                        "width": "92.4%",
                                                        "height": "8px",
                                                        "background": t["gradient"],
                                                        "borderRadius": "4px",
                                                    }
                                                ),
                                                style={
                                                    "background": t["border"],
                                                    "borderRadius": "4px",
                                                    "marginTop": "16px",
                                                },
                                            ),
                                        ]
                                    ),
                                ],
                                theme,
                            ),
                        ],
                        style={
                            "flex": "2",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
            ),
        ]
    )


# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
app.layout = html.Div(
    [
        dcc.Store(id="theme-store", data="cyberpunk"),
        dcc.Store(id="page-store", data="game"),
        dcc.Store(id="sessions-store", data=[]),
        dcc.Store(id="stats-store", data={}),
        dcc.Interval(id="refresh-interval", interval=5000, n_intervals=0),
        html.Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@400;700&family=IBM+Plex+Mono&family=VT323&family=Courier+Prime&family=Syne:wght@400;700;800&family=Space+Mono&display=swap",
        ),
        html.Div(
            id="main-container",
            children=[
                # ── SIDEBAR ──
                html.Div(
                    id="sidebar",
                    children=[
                        html.Div(
                            [
                                html.Img(
                                    src="/assets/logo_sise_gaming.png",
                                    style={
                                        "width": "100%",
                                        "maxWidth": "270px",
                                        "objectFit": "contain",
                                        "display": "block",
                                        "margin": "0 auto 28px auto",
                                        "filter": "drop-shadow(0 0 14px rgba(199,36,177,0.7))",
                                        "borderRadius": "12px",
                                    },
                                ),
                            ]
                        ),
                        html.Div(
                            "NAVIGATION",
                            style={
                                "fontSize": "9px",
                                "letterSpacing": "3px",
                                "opacity": "0.4",
                                "marginBottom": "12px",
                                "padding": "0 4px",
                            },
                        ),
                        html.Div(
                            [
                                html.Button(
                                    [
                                        html.Span("🎮", style={"marginRight": "10px"}),
                                        "Live Game",
                                    ],
                                    id="nav-game",
                                    n_clicks=0,
                                    className="nav-btn",
                                ),
                                html.Button(
                                    [
                                        html.Span("🧬", style={"marginRight": "10px"}),
                                        "Profils",
                                    ],
                                    id="nav-profils",
                                    n_clicks=0,
                                    className="nav-btn",
                                ),
                                html.Button(
                                    [
                                        html.Span("🎯", style={"marginRight": "10px"}),
                                        "Classifier",
                                    ],
                                    id="nav-classifier",
                                    n_clicks=0,
                                    className="nav-btn",
                                ),
                                html.Button(
                                    [
                                        html.Span("🤖", style={"marginRight": "10px"}),
                                        "Agent IA",
                                    ],
                                    id="nav-agent",
                                    n_clicks=0,
                                    className="nav-btn",
                                ),
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "4px",
                                "marginBottom": "32px",
                            },
                        ),
                        html.Div(
                            "THÈME",
                            style={
                                "fontSize": "9px",
                                "letterSpacing": "3px",
                                "opacity": "0.4",
                                "marginBottom": "12px",
                                "padding": "0 4px",
                            },
                        ),
                        html.Div(
                            [
                                html.Button(
                                    THEMES[th]["name"],
                                    id=f"theme-{th}",
                                    n_clicks=0,
                                    className="theme-btn",
                                    **{"data-theme": th},
                                )
                                for th in THEMES
                            ],
                            style={
                                "display": "flex",
                                "flexDirection": "column",
                                "gap": "4px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    "Master SISE 2025–2026",
                                    style={"fontSize": "10px", "opacity": "0.4"},
                                ),
                                html.Div(
                                    "Projet IA Temps réel",
                                    style={"fontSize": "10px", "opacity": "0.4"},
                                ),
                            ],
                            style={
                                "position": "absolute",
                                "bottom": "24px",
                                "left": "24px",
                            },
                        ),
                    ],
                ),
                html.Div(
                    id="page-content",
                    style={"flex": "1", "padding": "32px", "overflowY": "auto"},
                ),
            ],
        ),
    ],
    id="root",
)


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
@app.callback(
    Output("sessions-store", "data"),
    Output("stats-store", "data"),
    Input("refresh-interval", "n_intervals"),
)
def refresh_sessions(n):
    if not SUPABASE_AVAILABLE:
        return [], {}
    try:
        data = fetch_latest_sessions(limit=200)
        data = data if data else []
        stats = {
            "n_sessions": len(data),
            "n_players": len(set(d["player_name"] for d in data)) if data else 0,
            "avg_score": int(sum(d["score"] for d in data) / len(data)) if data else 0,
        }
        return data, stats
    except Exception:
        return [], {}


@app.callback(
    Output("theme-store", "data"),
    [Input(f"theme-{th}", "n_clicks") for th in THEMES],
    prevent_initial_call=True,
)
def update_theme(*args):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "cyberpunk"
    return ctx.triggered[0]["prop_id"].split(".")[0].replace("theme-", "")


@app.callback(
    Output("page-store", "data"),
    [
        Input("nav-game", "n_clicks"),
        Input("nav-profils", "n_clicks"),
        Input("nav-classifier", "n_clicks"),
        Input("nav-agent", "n_clicks"),
    ],
    prevent_initial_call=True,
)
def update_page(g, p, c, a):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "game"
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    return {
        "nav-game": "game",
        "nav-profils": "profils",
        "nav-classifier": "classifier",
        "nav-agent": "agent",
    }.get(btn, "game")


@app.callback(
    Output("main-container", "style"),
    Output("sidebar", "style"),
    Output("page-content", "children"),
    Input("theme-store", "data"),
    Input("page-store", "data"),
    dash.dependencies.State("sessions-store", "data"),
)
def render_all(theme, page, sessions_data):
    t = THEMES[theme]
    df_real = (
        pd.DataFrame(sessions_data)
        if sessions_data and len(sessions_data) >= 3
        else None
    )

    container_style = {
        "display": "flex",
        "minHeight": "100vh",
        "background": t["bg"],
        "color": t["text"],
        "fontFamily": t["font_body"],
    }
    sidebar_style = {
        "width": "220px",
        "minHeight": "100vh",
        "background": t["sidebar"],
        "borderRight": f"1px solid {t['border']}",
        "padding": "24px",
        "position": "relative",
        "fontFamily": t["font_body"],
    }
    pages = {
        "game": page_game(theme, df_real),
        "profils": page_profils(theme, df_real),
        "classifier": page_classifier(theme, df_real),
        "agent": page_agent(theme, df_real),
    }
    return container_style, sidebar_style, pages.get(page, page_game(theme, df_real))


@app.callback(
    Output("stat-sessions", "children"),
    Output("stat-players", "children"),
    Output("stat-score", "children"),
    Output("stat-reaction", "children"),
    Output("data-badge-container", "children"),
    Input("stats-store", "data"),
    Input("theme-store", "data"),
)
def update_stats(stats, theme):
    n_sess = str(stats.get("n_sessions", 0)) if stats else "—"
    n_players = str(stats.get("n_players", 0)) if stats else "—"
    avg_score = str(stats.get("avg_score", 0)) if stats else "—"
    is_real = bool(stats and stats.get("n_sessions", 0) > 0)
    return (
        stat_card(
            "Sessions", n_sess, "▲ Supabase live" if is_real else "En attente…", theme
        ).children,
        stat_card("Joueurs", n_players, "Uniques", theme).children,
        stat_card("Score moyen", avg_score, "Toutes sessions", theme).children,
        stat_card("Réaction moy.", "— ms", "Temps de réponse", theme).children,
        data_badge(is_real, theme),
    )


# @app.callback(
#     Output("live-inputs-table",  "children"),
#     Output("live-source-badge",  "children"),
#     Input("refresh-interval", "n_intervals"),
#     dash.dependencies.State("theme-store", "data"),
# )
# def update_live_inputs(n, theme):
#     t = THEMES[theme]
#     rows = []
#     try:
#         from core.supabase_client import fetch_live_inputs
#         rows = fetch_live_inputs(limit=10)
#     except Exception:
#         pass

#     # Badge source
#     source = rows[-1].get("event_type", "none") if rows else "none"
#     badge_map = {
#         "controller": ("🎮 Manette", t["accent2"]),
#         "keyboard":   ("⌨️  Clavier", t["accent3"]),
#     }
#     label, color = badge_map.get(source, ("⏳ En attente…", t["subtext"]))
#     badge = html.Div(label, style={"color": color, "fontSize": "10px", "letterSpacing": "1px",
#                                     "border": f"1px solid {color}", "borderRadius": "4px",
#                                     "padding": "2px 8px"})

#     return make_inputs_table(theme, rows), badge


@app.callback(
    Output("radar-graph", "figure"),
    Input("profile-selector", "value"),
    Input("theme-store", "data"),
)
def update_radar(profile, theme):
    return make_radar_fig(profile or "Précis", theme)


@app.callback(
    Output("launch-feedback", "children"),
    Output("launch-feedback", "style"),
    Input("btn-launch-game", "n_clicks"),
    dash.dependencies.State("input-player-name", "value"),
    dash.dependencies.State("dropdown-game-select", "value"),
    prevent_initial_call=True,
)
def launch_game(n_clicks, player_name, game_id):
    if not player_name or not player_name.strip():
        return "⚠️ Entre un nom de joueur avant de lancer.", {
            "marginTop": "12px",
            "fontSize": "12px",
            "color": "#FFB800",
        }
    if not game_id:
        return "⚠️ Sélectionne un jeu.", {
            "marginTop": "12px",
            "fontSize": "12px",
            "color": "#FFB800",
        }
    try:
        subprocess.Popen(
            [sys.executable, "main.py", game_id, player_name.strip()],
            cwd=ROOT_DIR,
        )
        return f"✅ '{game_id}' lancé pour {player_name} — bonne partie !", {
            "marginTop": "12px",
            "fontSize": "12px",
            "color": "#00F5FF",
        }
    except Exception as e:
        return f"❌ Erreur : {e}", {
            "marginTop": "12px",
            "fontSize": "12px",
            "color": "#FF4C6A",
        }


@app.callback(
    Output("live-inputs-table", "children"),
    Output("live-joystick-graph", "figure"),
    Output("gauge-lt", "children"),
    Output("gauge-rt", "children"),
    Output("live-buttons-display", "children"),
    Output("live-source-badge", "children"),
    Input("refresh-interval", "n_intervals"),
    dash.dependencies.State("theme-store", "data"),
)
def update_live_inputs(n, theme):
    t = THEMES[theme]
    rows = []
    try:
        from core.supabase_client import fetch_live_inputs

        rows = fetch_live_inputs(limit=60)
    except Exception:
        pass

    # ── Badge source ──
    source = rows[-1].get("event_type", "none") if rows else "none"
    badge_map = {
        "controller": ("🎮 Manette", t["accent2"]),
        "keyboard": ("⌨️  Clavier", t["accent3"]),
    }
    label, color = badge_map.get(source, ("⏳ En attente…", t["subtext"]))
    badge = html.Div(
        label,
        style={
            "color": color,
            "fontSize": "10px",
            "letterSpacing": "1px",
            "border": f"1px solid {color}",
            "borderRadius": "4px",
            "padding": "2px 8px",
        },
    )

    # ── Tableau (10 derniers) ──
    table = make_inputs_table(theme, rows[-10:] if rows else [])

    # ── Graphique joystick (60 derniers) ──
    if rows:
        lx_v = [r.get("lx", 0) for r in rows]
        ly_v = [r.get("ly", 0) for r in rows]
        ts = list(range(len(rows)))
    else:
        ts = list(range(60))
        lx_v = [np.sin(i * 0.2) * 0.5 for i in ts]
        ly_v = [np.cos(i * 0.15) * 0.4 for i in ts]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=lx_v,
            name="LX",
            line=dict(color=t["accent1"], width=2),
            mode="lines",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ts,
            y=ly_v,
            name="LY",
            line=dict(color=t["accent2"], width=2),
            mode="lines",
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor=t["border"], range=[-1.1, 1.1]),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1),
        margin=dict(l=20, r=20, t=10, b=10),
        height=180,
    )

    # ── Jauges ──
    last = rows[-1] if rows else {}
    lt_val = float(last.get("lt", 0) or 0)
    rt_val = float(last.get("rt", 0) or 0)
    gauge_lt = html.Div(
        style={
            "width": f"{int(lt_val*100)}%",
            "height": "8px",
            "background": t["accent2"],
            "borderRadius": "4px",
            "transition": "width 0.3s ease",
            "minWidth": "4px",
        }
    )
    gauge_rt = html.Div(
        style={
            "width": f"{int(rt_val*100)}%",
            "height": "8px",
            "background": t["accent1"],
            "borderRadius": "4px",
            "transition": "width 0.3s ease",
            "minWidth": "4px",
        }
    )

    # ── Boutons ──
    btns = {
        "A": last.get("btn_a", False),
        "B": last.get("btn_b", False),
        "X": last.get("btn_x", False),
        "Y": last.get("btn_y", False),
    }
    btn_colors = {"A": "#69FF47", "B": "#FF4C6A", "X": "#00E5FF", "Y": "#FFB800"}
    buttons_display = html.Div(
        [
            html.Div(
                btn,
                style={
                    "width": "36px",
                    "height": "36px",
                    "borderRadius": "50%",
                    "display": "flex",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "fontFamily": t["font"],
                    "fontSize": "12px",
                    "fontWeight": "700",
                    "background": btn_colors[btn] if pressed else t["border"],
                    "color": "#000" if pressed else t["subtext"],
                    "transition": "all 0.1s ease",
                    "boxShadow": f"0 0 10px {btn_colors[btn]}" if pressed else "none",
                },
            )
            for btn, pressed in btns.items()
        ],
        style={"display": "flex", "gap": "8px"},
    )

    return table, fig, gauge_lt, gauge_rt, buttons_display, badge


if __name__ == "__main__":
    app.run(debug=True, port=8050)
    # utiliser cette ligne si le port 8050 est réservé
    # app.run(debug=False, host="127.0.0.1", port=8200, use_reloader=False)
