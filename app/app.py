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
    "reflex":    "games/reflex_game.py",
    "labyrinth": "games/labyrinth_game.py",
    "shooter":   "games/shooter_game.py",
    "racing":    "games/racing_game.py",
}

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from core.supabase_client import fetch_all_sessions, fetch_latest_sessions
    SUPABASE_AVAILABLE = True
except Exception:
    SUPABASE_AVAILABLE = False

app = dash.Dash(
    __name__,
    assets_folder="assets",
    suppress_callback_exceptions=True,
    title="SISE Gaming — Controller Profiler"
)

# ─────────────────────────────────────────────
# MOCK DATA
# ─────────────────────────────────────────────
np.random.seed(42)
N = 60
cluster_centers    = {0: (2.0, 1.5), 1: (-2.0, 1.0), 2: (0.5, -2.5), 3: (-0.5, 2.5)}
cluster_names      = {0: "Agressif", 1: "Prudent", 2: "Précis", 3: "Chaotique"}
cluster_colors_map = {0: "#FF4C6A", 1: "#00E5FF", 2: "#69FF47", 3: "#FFB800"}

mock_players = [
    "Thomas","Emma","Lucas","Léa","Noah","Chloé","Ethan","Inès","Hugo",
    "Camille","Théo","Jade","Louis","Manon","Nathan","Alice","Axel",
    "Lucie","Maxime","Sarah","Raphaël"
]
umap_x, umap_y, labels, player_names = [], [], [], []
for i in range(N):
    c = i % 4
    cx, cy = cluster_centers[c]
    umap_x.append(cx + np.random.randn() * 0.6)
    umap_y.append(cy + np.random.randn() * 0.6)
    labels.append(c)
    player_names.append(mock_players[i % len(mock_players)])

df_umap_mock = pd.DataFrame({
    "x": umap_x, "y": umap_y,
    "cluster": [cluster_names[l] for l in labels],
    "player":  player_names,
})

features_list = ["Réactivité", "Agressivité", "Fluidité", "Précision", "Prise de risque", "Consistance"]
radar_profiles = {
    "Agressif":  [0.4, 0.95, 0.5, 0.6, 0.9, 0.5],
    "Prudent":   [0.8, 0.2,  0.75, 0.85, 0.2, 0.9],
    "Précis":    [0.9, 0.5,  0.9,  0.95, 0.5, 0.85],
    "Chaotique": [0.6, 0.8,  0.3,  0.4,  0.85, 0.3],
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
            num_cols  = ["btn_press_rate","lx_std","ly_std","reaction_time_avg_ms","input_regularity","score"]
            available = [c for c in num_cols if c in df_real.columns]
            X = df_real[available].fillna(0).values
            if X.shape[0] >= 3 and X.shape[1] >= 2:
                coords = PCA(n_components=2).fit_transform(StandardScaler().fit_transform(X))
                return pd.DataFrame({
                    "x": coords[:, 0], "y": coords[:, 1],
                    "cluster": df_real["game_id"].values,
                    "player":  df_real["player_name"].values,
                })
        except Exception as e:
            print(f"⚠️ PCA échoué : {e}")
    return df_umap_mock


# ─────────────────────────────────────────────
# THEMES
# ─────────────────────────────────────────────
THEMES = {
    "cyberpunk": {
        "name": " Cyberpunk",
        "bg": "#0A0A0F", "sidebar": "#0D0D1A", "card": "#12121F",
        "border": "#7B2FBE", "accent1": "#C724B1", "accent2": "#00F5FF",
        "accent3": "#FF4C6A", "text": "#E8E8FF", "subtext": "#8888AA",
        "font": "'Orbitron', monospace", "font_body": "'Share Tech Mono', monospace",
        "glow": "0 0 20px rgba(199,36,177,0.4)",
        "gradient": "linear-gradient(135deg, #C724B1 0%, #00F5FF 100%)",
    },
    "scientific": {
        "name": " Scientific",
        "bg": "#0B1120", "sidebar": "#0E1628", "card": "#111C35",
        "border": "#1E3A5F", "accent1": "#2979FF", "accent2": "#00BCD4",
        "accent3": "#FF6B35", "text": "#E3EAF4", "subtext": "#7A90B0",
        "font": "'Exo 2', sans-serif", "font_body": "'IBM Plex Mono', monospace",
        "glow": "0 0 20px rgba(41,121,255,0.3)",
        "gradient": "linear-gradient(135deg, #2979FF 0%, #00BCD4 100%)",
    },
    "matrix": {
        "name": " Matrix",
        "bg": "#030A03", "sidebar": "#050F05", "card": "#071207",
        "border": "#0D3B0D", "accent1": "#00FF41", "accent2": "#39FF14",
        "accent3": "#ADFF2F", "text": "#C8FFC8", "subtext": "#4A8A4A",
        "font": "'VT323', monospace", "font_body": "'Courier Prime', monospace",
        "glow": "0 0 20px rgba(0,255,65,0.4)",
        "gradient": "linear-gradient(135deg, #00FF41 0%, #39FF14 100%)",
    },
    "datasci": {
        "name": " DataSci",
        "bg": "#0F0E17", "sidebar": "#13121F", "card": "#1A1929",
        "border": "#2D2B45", "accent1": "#FF6B35", "accent2": "#F7C59F",
        "accent3": "#FFFFFE", "text": "#FFFFFE", "subtext": "#A7A9BE",
        "font": "'Syne', sans-serif", "font_body": "'Space Mono', monospace",
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
        "background": t["card"], "border": f"1px solid {t['border']}",
        "borderRadius": "12px", "padding": "20px", "boxShadow": t["glow"],
    }
    if style_extra:
        style.update(style_extra)
    return html.Div(children, style=style)


def stat_card(label, value, delta, theme):
    t = THEMES[theme]
    return make_card([
        html.Div(label, style={"color": t["subtext"], "fontSize": "11px",
                               "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "8px"}),
        html.Div(value, style={"color": t["accent1"], "fontSize": "28px",
                               "fontWeight": "700", "fontFamily": t["font"]}),
        html.Div(delta, style={"color": t["accent2"], "fontSize": "12px", "marginTop": "4px"}),
    ], theme, {"flex": "1", "minWidth": "140px"})


def data_badge(is_real, theme):
    t = THEMES[theme]
    if is_real:
        return html.Div("🟢 LIVE — Supabase", style={
            "color": t["accent2"], "fontSize": "10px", "letterSpacing": "2px",
            "border": f"1px solid {t['accent2']}", "borderRadius": "4px",
            "padding": "2px 8px", "display": "inline-block",
        })
    return html.Div("🟡 MOCK — En attente de sessions", style={
        "color": t["accent3"], "fontSize": "10px", "letterSpacing": "2px",
        "border": f"1px solid {t['accent3']}", "borderRadius": "4px",
        "padding": "2px 8px", "display": "inline-block",
    })


def make_inputs_table(theme, rows):
    """Tableau temps réel des derniers inputs — remplace le graphique simulé."""
    t = THEMES[theme]

    # Colonnes à afficher
    cols = ["#", "Joueur", "Jeu", "LX", "LY", "LT", "RT", "A", "B", "X", "Y", "Source"]

    header = html.Thead(html.Tr([
        html.Th(c, style={
            "color": t["subtext"], "fontSize": "10px", "padding": "6px 8px",
            "textTransform": "uppercase", "letterSpacing": "1px",
            "borderBottom": f"1px solid {t['border']}", "textAlign": "center",
            "whiteSpace": "nowrap",
        }) for c in cols
    ]))

    btn_colors = {"A": "#69FF47", "B": "#FF4C6A", "X": "#00E5FF", "Y": "#FFB800"}

    def btn_cell(val, color):
        active = bool(val)
        return html.Td(
            "●" if active else "○",
            style={
                "color": color if active else t["subtext"],
                "fontSize": "14px", "textAlign": "center", "padding": "5px 6px",
                "textShadow": f"0 0 6px {color}" if active else "none",
            }
        )

    def val_cell(v, accent=False):
        try:
            fv = float(v)
            color = t["accent1"] if accent else (t["accent2"] if fv > 0.3 else t["text"])
        except Exception:
            color = t["text"]
        return html.Td(
            f"{float(v):.2f}" if v not in (None, "", "—") else "—",
            style={"color": color, "fontSize": "11px", "textAlign": "center",
                   "padding": "5px 8px", "fontFamily": t["font_body"]},
        )

    if not rows:
        # Lignes mock pour montrer la structure
        mock_rows = [
            {"idx": i+1, "player": "—", "game": "—",
             "lx": 0.0, "ly": 0.0, "lt": 0.0, "rt": 0.0,
             "btn_a": False, "btn_b": False, "btn_x": False, "btn_y": False,
             "source": "—"}
            for i in range(5)
        ]
    else:
        mock_rows = [
            {"idx": len(rows) - i,
             "player": r.get("player_name", "—"),
             "game":   r.get("game_id", "—"),
             "lx":  r.get("lx", 0), "ly":  r.get("ly", 0),
             "lt":  r.get("lt", 0), "rt":  r.get("rt", 0),
             "btn_a": r.get("btn_a", False), "btn_b": r.get("btn_b", False),
             "btn_x": r.get("btn_x", False), "btn_y": r.get("btn_y", False),
             "source": r.get("event_type", "—")}
            for i, r in enumerate(reversed(rows[-10:]))
        ]

    game_colors = {
        "reflex": "#FF4C6A", "labyrinth": "#00E5FF",
        "shooter": "#69FF47", "racing": "#FFB800",
    }
    source_icons = {"controller": "🎮", "keyboard": "⌨️", "—": "⏳"}

    body_rows = []
    for row in mock_rows:
        gc = game_colors.get(row["game"], t["subtext"])
        body_rows.append(html.Tr([
            html.Td(str(row["idx"]), style={"color": t["subtext"], "fontSize": "10px",
                                             "textAlign": "center", "padding": "5px 8px"}),
            html.Td(row["player"], style={"color": t["text"], "fontSize": "11px",
                                           "padding": "5px 8px", "fontFamily": t["font_body"]}),
            html.Td(row["game"], style={"color": gc, "fontSize": "11px",
                                         "padding": "5px 8px", "fontFamily": t["font_body"]}),
            val_cell(row["lx"]),
            val_cell(row["ly"]),
            val_cell(row["lt"], accent=True),
            val_cell(row["rt"], accent=True),
            btn_cell(row["btn_a"], btn_colors["A"]),
            btn_cell(row["btn_b"], btn_colors["B"]),
            btn_cell(row["btn_x"], btn_colors["X"]),
            btn_cell(row["btn_y"], btn_colors["Y"]),
            html.Td(source_icons.get(row["source"], "⏳"),
                    style={"textAlign": "center", "padding": "5px 8px", "fontSize": "13px"}),
        ], style={"borderBottom": f"1px solid {t['border']}",
                  "transition": "background 0.2s"}))

    return html.Table(
        [header, html.Tbody(body_rows)],
        style={"width": "100%", "borderCollapse": "collapse", "tableLayout": "fixed"}
    )


# ─────────────────────────────────────────────
# FIGURES
# ─────────────────────────────────────────────
def make_umap_fig(theme, df_real=None):
    t = THEMES[theme]
    plot_df = build_umap_df(df_real)
    colors_by_group = {
        "reflex": "#FF4C6A", "labyrinth": "#00E5FF",
        "shooter": "#69FF47", "racing": "#FFB800",
        "Agressif": "#FF4C6A", "Prudent": "#00E5FF",
        "Précis": "#69FF47", "Chaotique": "#FFB800",
    }
    fig = go.Figure()
    for group in plot_df["cluster"].unique():
        mask  = plot_df["cluster"] == group
        color = colors_by_group.get(group, "#AAAAAA")
        fig.add_trace(go.Scatter(
            x=plot_df[mask]["x"], y=plot_df[mask]["y"],
            mode="markers+text", name=str(group),
            text=plot_df[mask]["player"], textposition="top center",
            textfont=dict(size=8, color=color),
            marker=dict(size=10, color=color, opacity=0.85, line=dict(width=1, color="white")),
            hovertemplate="<b>%{text}</b><br>Groupe: " + str(group) + "<extra></extra>",
        ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=t["border"], borderwidth=1),
        xaxis=dict(showgrid=True, gridcolor=t["border"], zeroline=False, title="Axe 1"),
        yaxis=dict(showgrid=True, gridcolor=t["border"], zeroline=False, title="Axe 2"),
        margin=dict(l=20, r=20, t=20, b=20), height=380,
    )
    return fig


def make_radar_fig(profile_name, theme):
    t    = THEMES[theme]
    vals = radar_profiles.get(profile_name, radar_profiles["Précis"])
    fig  = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=features_list + [features_list[0]],
        fill="toself",
        fillcolor=f"rgba({int(t['accent1'][1:3],16)},{int(t['accent1'][3:5],16)},{int(t['accent1'][5:7],16)},0.25)",
        line=dict(color=t["accent1"], width=2), name=profile_name,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,1], gridcolor=t["border"], color=t["subtext"]),
            angularaxis=dict(gridcolor=t["border"], color=t["text"]),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        margin=dict(l=30, r=30, t=30, b=30), height=300, showlegend=False,
    )
    return fig


def make_reaction_hist(theme, df_real=None):
    t    = THEMES[theme]
    data = None
    if df_real is not None and "reaction_time_avg_ms" in df_real.columns:
        tmp = df_real["reaction_time_avg_ms"].dropna().values
        if len(tmp[tmp > 0]) > 0:
            data = tmp[tmp > 0]
    if data is None or len(data) == 0:
        data = np.concatenate([
            np.random.normal(180, 20, 30),
            np.random.normal(240, 30, 25),
            np.random.normal(310, 25, 20),
        ])
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, nbinsx=25, marker_color=t["accent1"], opacity=0.8))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        xaxis=dict(showgrid=False, title="Réaction (ms)"),
        yaxis=dict(showgrid=True, gridcolor=t["border"]),
        margin=dict(l=20, r=20, t=10, b=20), height=220,
        showlegend=False, bargap=0.05,
    )
    return fig


def make_score_bar(theme, df_real=None):
    t = THEMES[theme]
    if df_real is not None and "score" in df_real.columns:
        grp    = df_real.groupby("game_id")["score"].mean().reset_index()
        games  = grp["game_id"].tolist()
        scores = grp["score"].tolist()
    else:
        games  = ["reflex", "labyrinth", "shooter", "racing"]
        scores = [620, 850, 1200, 2400]
    colors = ["#FF4C6A", "#00E5FF", "#69FF47", "#FFB800"]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=games, y=scores, marker_color=colors[:len(games)], opacity=0.85))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=t["border"]),
        margin=dict(l=20, r=20, t=10, b=20), height=200, showlegend=False,
    )
    return fig


def make_agent_comparison(theme):
    t     = THEMES[theme]
    cats  = ["Réactivité", "Précision", "Fluidité", "Agressivité", "Consistance"]
    human = [0.72, 0.65, 0.80, 0.55, 0.68]
    agent = [0.69, 0.67, 0.77, 0.58, 0.71]
    fig   = go.Figure()
    fig.add_trace(go.Bar(name="Humain",   x=cats, y=human, marker_color=t["accent2"], opacity=0.85))
    fig.add_trace(go.Bar(name="Agent IA", x=cats, y=agent, marker_color=t["accent1"], opacity=0.85))
    fig.update_layout(
        barmode="group", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=t["border"], range=[0, 1]),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=20, r=20, t=10, b=20), height=250,
    )
    return fig


# ─────────────────────────────────────────────
# PAGES
# ─────────────────────────────────────────────
def page_game(theme, df_real=None):
    t       = THEMES[theme]
    is_real = df_real is not None

    return html.Div([
        # ── Titre + badge ──
        html.Div([
            html.Div([
                html.Div("🎮 Session Live", style={"color": t["accent1"], "fontSize": "22px",
                                                    "fontWeight": "700", "fontFamily": t["font"], "marginBottom": "4px"}),
                html.Div("Capture des inputs manette en temps réel",
                         style={"color": t["subtext"], "fontSize": "13px"}),
            ]),
            html.Div(id="data-badge-container"),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "flex-start", "marginBottom": "24px"}),

        # ── Métriques ──
        html.Div([
            html.Div(id="stat-sessions", style={"flex": "1", "minWidth": "140px"}),
            html.Div(id="stat-players",  style={"flex": "1", "minWidth": "140px"}),
            html.Div(id="stat-score",    style={"flex": "1", "minWidth": "140px"}),
            html.Div(id="stat-reaction", style={"flex": "1", "minWidth": "140px"}),
        ], style={"display": "flex", "gap": "16px", "marginBottom": "24px", "flexWrap": "wrap"}),

        # ── Lancer un jeu (remonté ici) ──
        make_card([
            html.Div("Lancer un jeu", style={"color": t["subtext"], "fontSize": "11px",
                                              "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "16px"}),
            html.Div([
                html.Div([
                    html.Div("Nom du joueur", style={"color": t["subtext"], "fontSize": "11px", "marginBottom": "6px"}),
                    dcc.Input(
                        id="input-player-name",
                        placeholder="ex: Thomas",
                        debounce=True,
                        style={
                            "background": t["bg"], "border": f"1px solid {t['border']}",
                            "color": t["text"], "padding": "8px 12px", "borderRadius": "6px",
                            "fontFamily": t["font_body"], "width": "180px",
                        }
                    ),
                ]),
                html.Div([
                    html.Div("Jeu", style={"color": t["subtext"], "fontSize": "11px", "marginBottom": "6px"}),
                    dcc.Dropdown(
                        id="dropdown-game-select",
                        options=[
                            {"label": " Reflex",    "value": "reflex"},
                            {"label": " Labyrinth", "value": "labyrinth"},
                            {"label": " Shooter",   "value": "shooter"},
                            {"label": " Racing",    "value": "racing"},
                        ],
                        value="reflex", clearable=False,
                        style={"background": t["bg"], "color": "#000",
                               "border": f"1px solid {t['border']}", "borderRadius": "6px",
                               "width": "200px", "fontFamily": t["font_body"]},
                    ),
                ]),
                html.Button("▶ LANCER LE JEU", id="btn-launch-game", n_clicks=0, style={
                    "background": t["gradient"], "border": "none", "color": "#000",
                    "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer",
                    "fontFamily": t["font"], "fontSize": "13px", "fontWeight": "700",
                    "letterSpacing": "2px", "alignSelf": "flex-end",
                }),
            ], style={"display": "flex", "gap": "16px", "alignItems": "flex-end", "flexWrap": "wrap"}),
            html.Div(id="launch-feedback", style={"marginTop": "12px", "fontSize": "12px"}),
        ], theme, {"marginBottom": "16px"}),

        make_card([
    html.Div([
        html.Div("Flux inputs temps réel", style={"color": t["subtext"], "fontSize": "11px",
                                                    "textTransform": "uppercase", "letterSpacing": "2px"}),
        html.Div(id="live-source-badge"),  # affiche "🎮 Manette" ou "⌨️ Clavier"
    ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "12px"}),

    # Graphique joysticks
    dcc.Graph(id="live-joystick-graph", config={"displayModeBar": False},
              style={"height": "180px"}),

    # Jauges gâchettes
    html.Div([
        html.Div([
            html.Div("LT", style={"color": t["subtext"], "fontSize": "11px", "marginBottom": "4px"}),
            html.Div(id="gauge-lt", style={"background": t["border"], "borderRadius": "4px", "height": "8px"}),
        ], style={"flex": "1"}),
        html.Div([
            html.Div("RT", style={"color": t["subtext"], "fontSize": "11px", "marginBottom": "4px"}),
            html.Div(id="gauge-rt", style={"background": t["border"], "borderRadius": "4px", "height": "8px"}),
        ], style={"flex": "1"}),
    ], style={"display": "flex", "gap": "16px", "marginTop": "12px", "marginBottom": "12px"}),

    # Boutons
    html.Div(id="live-buttons-display"),

], theme, {"flex": "1"}),

        # ── Flux inputs + Distribution réactions ──
        html.Div([
            # Flux inputs temps réel → tableau
            make_card([
                html.Div([
                    html.Div("Inputs manette", style={"color": t["subtext"], "fontSize": "11px",
                                                       "textTransform": "uppercase", "letterSpacing": "2px"}),
                    html.Div(id="live-source-badge"),
                ], style={"display": "flex", "justifyContent": "space-between",
                          "alignItems": "center", "marginBottom": "12px"}),
                html.Div(id="live-inputs-table",
                         style={"overflowX": "auto", "overflowY": "auto", "maxHeight": "280px"}),
            ], theme, {"flex": "1"}),

            # Distribution réactions
            # html.Div([
            #     make_card([
            #         html.Div("Distribution réactions", style={"color": t["subtext"], "fontSize": "11px",
            #                                                    "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "12px"}),
            #         dcc.Graph(figure=make_reaction_hist(theme, df_real), config={"displayModeBar": False}),
            #     ], theme),
            # ], style={"flex": "1", "display": "flex", "flexDirection": "column"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"}),

        # ── Score moyen par jeu ──
        make_card([
                    html.Div("Distribution réactions", style={"color": t["subtext"], "fontSize": "11px",
                                                               "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "12px"}),
                    dcc.Graph(figure=make_reaction_hist(theme, df_real), config={"displayModeBar": False}),
                ], theme),

    ])


# ── Cache analyse (évite recalcul à chaque changement de thème) ──────────────
_profils_cache: dict = {}


def _get_shooter_analysis(df_real):
    """Retourne l'analyse mise en cache si les données n'ont pas changé."""
    if df_real is None or len(df_real) < 5:
        return None
    try:
        key = f"{len(df_real)}_{df_real['score'].sum() if 'score' in df_real.columns else 0}"
        if key not in _profils_cache:
            sys.path.insert(0, ROOT_DIR)
            from analysis_shooter import compute_shooter_analysis
            _profils_cache.clear()
            _profils_cache[key] = compute_shooter_analysis(df_real)
        return _profils_cache[key]
    except Exception as e:
        print(f"⚠️  Analyse shooter échouée : {e}")
        return None


_CLUSTER_COLORS_HEX = ["#C724B1", "#00F5FF", "#69FF47", "#FFB800"]
_STATUS_COLORS = {
    "📈 En progression": "#69FF47",
    "➡️ Stable":         "#00F5FF",
    "📉 En régression":  "#FF4C6A",
}
_FEAT_LABELS = {
    "btn_press_rate": "Pression boutons", "btn_variety": "Variété boutons",
    "btn_hold_avg_ms": "Durée appui (ms)", "lx_mean": "Joystick G. (X)",
    "ly_mean": "Joystick G. (Y)", "lx_std": "Agitation G.",
    "ly_std": "Agitation G. (Y)", "lx_direction_changes": "Chgt direction",
    "rx_std": "Agitation visée (X)", "ry_std": "Agitation visée (Y)",
    "rt_mean": "Gâchette tir", "lt_mean": "Gâchette gauche",
    "input_regularity": "Régularité",
}


def _profils_no_data(theme, msg="Pas assez de sessions shooter pour l'analyse."):
    t = THEMES[theme]
    return html.Div([
        html.Div("🧬 Analyse Comportementale · Shooter", style={
            "color": t["accent1"], "fontSize": "22px",
            "fontWeight": "700", "fontFamily": t["font"], "marginBottom": "16px",
        }),
        make_card([
            html.Div("⚠️  " + msg, style={"color": t["subtext"], "fontSize": "14px",
                                            "textAlign": "center", "padding": "40px"}),
            html.Div("Jouez des parties en mode 'shooter' pour alimenter l'analyse.",
                     style={"color": t["subtext"], "fontSize": "12px", "textAlign": "center"}),
        ], theme),
    ])


def _make_plotly_cfg():
    return {"displayModeBar": False}


def page_profils(theme, df_real=None, active_tab="clustering"):
    t      = THEMES[theme]
    result = _get_shooter_analysis(df_real)

    if result is None:
        n_shooter = 0
        if df_real is not None and "game_id" in df_real.columns:
            n_shooter = (df_real["game_id"] == "shooter").sum()
        if n_shooter > 0:
            return _profils_no_data(theme, f"Seulement {n_shooter} session(s) shooter — minimum 5 requis.")
        return _profils_no_data(theme)

    cl = result["clustering"]
    pr = result["progression"]
    co = result["correlation"]
    df_sh = result["df"]
    is_real = df_real is not None

    bg   = t["bg"]
    card = t["card"]
    brd  = t["border"]
    txt  = t["text"]
    sub  = t["subtext"]
    plot_base = dict(
        paper_bgcolor=card, plot_bgcolor=card,
        font=dict(color=txt, family=t["font_body"], size=10),
        margin=dict(l=10, r=10, t=36, b=10),
    )

    # ── KPI strip ──────────────────────────────────────────────────────────────
    n_players = df_sh["player_name"].nunique() if "player_name" in df_sh.columns else "?"
    top_feat  = _FEAT_LABELS.get(co["top3"][0], co["top3"][0]) if co["top3"] else "—"
    best_k    = cl.get("best_k", 3)
    kpi_strip = html.Div([
        stat_card("Joueurs",     str(n_players), "Analysés",        theme),
        stat_card("Sessions",    str(cl["n"]),   "Shooter",         theme),
        stat_card("Clusters",    "3",            f"Optimal : k={best_k}", theme),
        stat_card("Top feature", top_feat,       "Prédicteur score", theme),
    ], style={"display": "flex", "gap": "12px", "marginBottom": "20px", "flexWrap": "wrap"})

    # ══════════════════════════════════════════════════════════════════════════
    # ANALYSE 1 — Clustering
    # ══════════════════════════════════════════════════════════════════════════
    coords     = cl["umap_xy"] if cl["umap_xy"] is not None else cl["pca_xy"]
    proj_label = "UMAP" if cl["umap_xy"] is not None else "PCA"

    # Scatter par cluster
    traces_cl = []
    for c, name in cl["cluster_names"].items():
        mask = cl["labels"] == c
        col  = _CLUSTER_COLORS_HEX[c % 4]
        traces_cl.append(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers", name=name,
            text=cl["player_names"][mask],
            hovertemplate="<b>%{text}</b><br>" + proj_label + " : (%{x:.2f}, %{y:.2f})<extra>" + name + "</extra>",
            marker=dict(color=col, size=10, opacity=0.85,
                        line=dict(color=bg, width=1)),
        ))
    fig_cl = go.Figure(traces_cl)
    fig_cl.update_layout(**plot_base,
                          title=dict(text=f"Projection {proj_label} · colorée par cluster",
                                     font=dict(size=11, color=sub)),
                          legend=dict(orientation="h", y=-0.15, x=0),
                          height=340, showlegend=True,
                          xaxis=dict(gridcolor=brd, zeroline=False),
                          yaxis=dict(gridcolor=brd, zeroline=False))

    # Scatter par joueur
    players_u = sorted(set(cl["player_names"]))
    import plotly.colors as pc
    pal = pc.qualitative.Plotly + pc.qualitative.D3
    traces_pl = []
    for i, player in enumerate(players_u):
        mask = cl["player_names"] == player
        traces_pl.append(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers", name=player,
            text=[cl["cluster_names"].get(int(c), str(c)) for c in cl["labels"][mask]],
            hovertemplate="<b>" + player + "</b><br>Cluster : %{text}<extra></extra>",
            marker=dict(color=pal[i % len(pal)], size=9, opacity=0.85,
                        line=dict(color=bg, width=1),
                        symbol="circle"),
        ))
    fig_pl = go.Figure(traces_pl)
    fig_pl.update_layout(**plot_base,
                          title=dict(text=f"Projection {proj_label} · colorée par joueur",
                                     font=dict(size=11, color=sub)),
                          legend=dict(orientation="h", y=-0.3, x=0, font=dict(size=9)),
                          height=340, showlegend=True,
                          xaxis=dict(gridcolor=brd, zeroline=False),
                          yaxis=dict(gridcolor=brd, zeroline=False))

    # Silhouette scores
    sil_k    = list(cl["silhouette_scores"].keys())
    sil_vals = list(cl["silhouette_scores"].values())
    sil_cols = [_CLUSTER_COLORS_HEX[i % 4] for i in range(len(sil_k))]
    fig_sil  = go.Figure(go.Bar(
        x=[str(k) for k in sil_k], y=sil_vals,
        marker=dict(color=sil_cols, line=dict(width=0)),
        text=[f"{v:.3f}" for v in sil_vals], textposition="outside",
        textfont=dict(size=11, family=t["font"]),
        hovertemplate="k=%{x}<br>Silhouette=%{y:.4f}<extra></extra>",
    ))
    fig_sil.add_shape(type="line", x0=str(3), x1=str(3), y0=0,
                      y1=max(sil_vals) * 1.1 if sil_vals else 1,
                      line=dict(color="#FFB800", dash="dash", width=2))
    fig_sil.update_layout(**plot_base,
                           title=dict(text="Silhouette Score · k=2,3,4",
                                      font=dict(size=11, color=sub)),
                           height=240,
                           xaxis=dict(title="k clusters", gridcolor=brd),
                           yaxis=dict(title="Silhouette", gridcolor=brd, range=[0, max(sil_vals) * 1.2] if sil_vals else [0, 1]),
                           showlegend=False)

    # Heatmap centroïdes normalisés
    cn      = cl["centroids_norm"]
    c_names = [cl["cluster_names"].get(i, str(i)) for i in range(len(cn))]
    f_labs  = [_FEAT_LABELS.get(f, f) for f in cn.columns]
    fig_heat = go.Figure(go.Heatmap(
        z=cn.values, x=f_labs, y=c_names,
        colorscale="RdBu_r", zmid=0, zmin=-2.5, zmax=2.5,
        text=np.round(cn.values, 2),
        texttemplate="%{text}", textfont=dict(size=9),
        hovertemplate="<b>%{y}</b><br>%{x}<br>z-score : %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=12, len=0.8, tickfont=dict(size=9)),
    ))
    fig_heat.update_layout(**plot_base,
                            title=dict(text="Centroïdes normalisés — rouge=élevé, bleu=faible",
                                       font=dict(size=11, color=sub)),
                            height=200,
                            xaxis=dict(tickangle=-35, tickfont=dict(size=9)),
                            yaxis=dict(tickfont=dict(size=10)))

    # Tableau centroides réels
    cr       = cl["centroids_real"]
    cr_cols  = list(cr.columns)
    cr_heads = [_FEAT_LABELS.get(c, c) for c in cr_cols]
    tbl_cl_rows = []
    for i, row in cr.iterrows():
        c_col = _CLUSTER_COLORS_HEX[int(i) % 4]
        cells = []
        for j, val in enumerate(row):
            is_name = (cr_cols[j] in ["Profil", "Sessions"])
            cells.append(html.Td(
                str(val),
                style={"padding": "7px 10px",
                       "borderBottom": f"1px solid {brd}",
                       "color": c_col if j == 0 else (txt if is_name else sub),
                       "fontFamily": t["font"] if j == 0 else t["font_body"],
                       "fontWeight": "700" if j == 0 else "400",
                       "fontSize": "12px", "whiteSpace": "nowrap"}
            ))
        tbl_cl_rows.append(html.Tr(cells))

    tbl_cl = html.Table([
        html.Thead(html.Tr([
            html.Th(h, style={"padding": "7px 10px", "color": sub, "fontSize": "10px",
                               "textTransform": "uppercase", "letterSpacing": "1px",
                               "borderBottom": f"2px solid {brd}", "whiteSpace": "nowrap"})
            for h in cr_heads
        ])),
        html.Tbody(tbl_cl_rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})

    tab_clustering = html.Div([
        html.Div([
            html.Div([
                make_card([dcc.Graph(figure=fig_cl, config=_make_plotly_cfg())],
                           theme, {"flex": "1", "minWidth": "320px"}),
                make_card([dcc.Graph(figure=fig_pl, config=_make_plotly_cfg())],
                           theme, {"flex": "1", "minWidth": "320px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                make_card([dcc.Graph(figure=fig_sil, config=_make_plotly_cfg())],
                           theme, {"flex": "1", "minWidth": "240px"}),
                make_card([
                    html.Div("Caractéristiques moyennes par cluster", style={
                        "color": sub, "fontSize": "10px", "textTransform": "uppercase",
                        "letterSpacing": "2px", "marginBottom": "10px",
                    }),
                    html.Div(tbl_cl, style={"overflowX": "auto"}),
                ], theme, {"flex": "2", "minWidth": "340px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            make_card([dcc.Graph(figure=fig_heat, config=_make_plotly_cfg())], theme),
        ])
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # ANALYSE 2 — Progression
    # ══════════════════════════════════════════════════════════════════════════
    players = pr["players"]

    if not players:
        tab_progression = make_card([
            html.Div("Pas assez de sessions par joueur pour calculer la progression.",
                     style={"color": sub, "textAlign": "center", "padding": "30px"})
        ], theme)
    else:
        # Synthèse : barplot des pentes
        slope_cols = [_STATUS_COLORS.get(s, t["accent1"]) for s in pr["statuses"]]
        fig_slopes = go.Figure(go.Bar(
            x=pr["slopes"].tolist(), y=players,
            orientation="h",
            marker=dict(color=slope_cols, line=dict(width=0)),
            text=[f"{s:+.2f}" for s in pr["slopes"]],
            textposition="outside",
            textfont=dict(size=10, family=t["font"]),
            hovertemplate="<b>%{y}</b><br>Pente : %{x:+.2f} pts/session<extra></extra>",
        ))
        fig_slopes.add_vline(x=0, line=dict(color=sub, width=1, dash="dash"))
        fig_slopes.update_layout(
            **plot_base,
            title=dict(text="Pente de progression par joueur (pts/session)", font=dict(size=11, color=sub)),
            height=max(280, len(players) * 28 + 60),
            xaxis=dict(gridcolor=brd, zeroline=False),
            yaxis=dict(gridcolor=brd, autorange="reversed"),
            showlegend=False,
        )

        # Grille : courbes individuelles (max 12 joueurs affichés)
        shown_players = players[:12]
        ncols_g = min(4, len(shown_players))
        nrows_g = (len(shown_players) + ncols_g - 1) // ncols_g

        from plotly.subplots import make_subplots
        player_titles = [
            f"{p}<br><span style='font-size:9px'>{pr['statuses'][players.index(p)]}</span>"
            for p in shown_players
        ]
        fig_grid = make_subplots(
            rows=nrows_g, cols=ncols_g,
            subplot_titles=player_titles,
            horizontal_spacing=0.06, vertical_spacing=0.14,
        )
        fig_grid.update_layout(**plot_base, height=nrows_g * 200 + 60, showlegend=False)

        for i, player in enumerate(shown_players):
            r, c = divmod(i, ncols_g)
            idx = players.index(player)
            data = pr["sessions_data"].get(player, [])
            if not data:
                continue
            xs, ys = zip(*data)
            s_col = _STATUS_COLORS.get(pr["statuses"][idx], t["accent1"])
            fig_grid.add_trace(go.Scatter(
                x=list(xs), y=list(ys), mode="lines+markers", name=player,
                line=dict(color=s_col, width=2),
                marker=dict(color=s_col, size=6, line=dict(color=bg, width=1)),
                hovertemplate=f"<b>{player}</b><br>Session %{{x}}<br>Score : %{{y}}<extra></extra>",
            ), row=r + 1, col=c + 1)
            tx, ty = pr["trend_lines"].get(player, (xs, ys))
            fig_grid.add_trace(go.Scatter(
                x=list(tx), y=list(ty), mode="lines", showlegend=False,
                line=dict(color="#FFB800", width=1.5, dash="dash"),
            ), row=r + 1, col=c + 1)

        # Couleur des sous-titres
        for i, ann in enumerate(fig_grid.layout.annotations[:len(shown_players)]):
            idx = players.index(shown_players[i])
            ann.update(font=dict(size=10, color=_STATUS_COLORS.get(pr["statuses"][idx], sub)))

        # Grilles et axes
        for r in range(1, nrows_g + 1):
            for c in range(1, ncols_g + 1):
                fig_grid.update_xaxes(gridcolor=brd, zeroline=False, row=r, col=c)
                fig_grid.update_yaxes(gridcolor=brd, zeroline=False, row=r, col=c)

        # Tableau statut
        status_rows = []
        for p, st, sl, r2, pv in zip(pr["players"], pr["statuses"], pr["slopes"],
                                      pr["r2"], pr["pvalues"]):
            st_col = _STATUS_COLORS.get(st, sub)
            status_rows.append(html.Tr([
                html.Td(p,       style={"padding": "6px 10px", "borderBottom": f"1px solid {brd}", "color": txt, "fontSize": "12px"}),
                html.Td(st,      style={"padding": "6px 10px", "borderBottom": f"1px solid {brd}", "color": st_col, "fontSize": "12px"}),
                html.Td(f"{sl:+.2f}", style={"padding": "6px 10px", "borderBottom": f"1px solid {brd}", "color": st_col, "fontSize": "12px", "fontFamily": t["font"]}),
                html.Td(f"{r2:.3f}", style={"padding": "6px 10px", "borderBottom": f"1px solid {brd}", "color": sub, "fontSize": "12px"}),
                html.Td(f"{pv:.3f}", style={"padding": "6px 10px", "borderBottom": f"1px solid {brd}", "color": "#FFB800" if pv < 0.1 else sub, "fontSize": "12px"}),
            ]))

        tbl_prog = html.Table([
            html.Thead(html.Tr([
                html.Th(h, style={"padding": "6px 10px", "color": sub, "fontSize": "10px",
                                   "textTransform": "uppercase", "letterSpacing": "1px",
                                   "borderBottom": f"2px solid {brd}"})
                for h in ["Joueur", "Statut", "Pente β", "R²", "p-value"]
            ])),
            html.Tbody(status_rows),
        ], style={"width": "100%", "borderCollapse": "collapse"})

        tab_progression = html.Div([
            html.Div([
                make_card([dcc.Graph(figure=fig_slopes, config=_make_plotly_cfg())],
                           theme, {"flex": "1", "minWidth": "280px", "marginBottom": "12px"}),
                make_card([
                    html.Div("Résumé statistique", style={
                        "color": sub, "fontSize": "10px", "textTransform": "uppercase",
                        "letterSpacing": "2px", "marginBottom": "10px",
                    }),
                    tbl_prog,
                ], theme, {"flex": "1", "minWidth": "280px", "marginBottom": "12px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
            make_card([
                html.Div("Courbes de score par joueur (avec droite de tendance)",
                         style={"color": sub, "fontSize": "10px", "textTransform": "uppercase",
                                "letterSpacing": "2px", "marginBottom": "10px"}),
                dcc.Graph(figure=fig_grid, config=_make_plotly_cfg()),
            ], theme),
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # ANALYSE 3 — Corrélation
    # ══════════════════════════════════════════════════════════════════════════
    if not co["features"] or "score" not in df_sh.columns:
        tab_correlation = make_card([
            html.Div("Données insuffisantes.", style={"color": sub, "textAlign": "center", "padding": "30px"})
        ], theme)
    else:
        feats_lab = [_FEAT_LABELS.get(f, f) for f in co["features"]]
        rvals     = co["spearman_r"].tolist()
        pvals     = co["pvalues"].tolist()
        bar_cols  = [t["accent1"] if r >= 0 else t["accent3"] for r in rvals]
        stars     = ["***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "")) for p in pvals]

        fig_corr = go.Figure(go.Bar(
            x=rvals[::-1],
            y=feats_lab[::-1],
            orientation="h",
            marker=dict(color=bar_cols[::-1], line=dict(width=0),
                        opacity=[0.95 if abs(r) > 0.3 else 0.5 for r in rvals[::-1]]),
            text=[f"{r:+.3f} {s}" for r, s in zip(rvals[::-1], stars[::-1])],
            textposition="outside",
            textfont=dict(size=10, family=t["font"]),
            hovertemplate="<b>%{y}</b><br>Spearman r : %{x:+.3f}<br>p : " +
                          "<extra></extra>",
        ))
        fig_corr.add_vline(x=0, line=dict(color=sub, width=1))
        fig_corr.add_vrect(x0=-0.3, x1=0.3, fillcolor=sub, opacity=0.05, line_width=0)
        fig_corr.update_layout(
            **plot_base,
            title=dict(text="Corrélation Spearman avec le Score  (* p<.05  ** p<.01  *** p<.001)",
                       font=dict(size=11, color=sub)),
            height=max(320, len(feats_lab) * 26 + 80),
            xaxis=dict(gridcolor=brd, zeroline=False, range=[-1.1, 1.1]),
            yaxis=dict(gridcolor=brd),
            showlegend=False,
        )

        # Scatter top 2 features vs score
        scatter_figs = []
        for feat_i in range(min(2, len(co["features"]))):
            feat  = co["features"][feat_i]
            label = _FEAT_LABELS.get(feat, feat)
            x_sc  = df_sh[feat].fillna(0).values
            y_sc  = df_sh["score"].fillna(0).values
            players_sc = df_sh["player_name"].values if "player_name" in df_sh.columns else ["?"] * len(x_sc)
            r_val = co["spearman_r"][feat_i]
            p_val = co["pvalues"][feat_i]

            m, b_int = np.polyfit(x_sc, y_sc, 1)
            x_line = np.linspace(x_sc.min(), x_sc.max(), 80)

            f_scat = go.Figure([
                go.Scatter(
                    x=x_sc.tolist(), y=y_sc.tolist(), mode="markers",
                    text=players_sc.tolist(),
                    hovertemplate="<b>%{text}</b><br>" + label + " : %{x:.3f}<br>Score : %{y}<extra></extra>",
                    marker=dict(color=t["accent1"], size=9, opacity=0.8,
                                line=dict(color=bg, width=1)),
                ),
                go.Scatter(
                    x=x_line.tolist(), y=(m * x_line + b_int).tolist(),
                    mode="lines", showlegend=False,
                    line=dict(color="#FFB800", width=2, dash="dash"),
                ),
            ])
            star_s = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
            f_scat.update_layout(
                **plot_base,
                title=dict(text=f"{label}  vs  Score  (r={r_val:+.3f}, p={p_val:.3f} {star_s})",
                           font=dict(size=11, color=sub)),
                height=280,
                xaxis=dict(title=label, gridcolor=brd, zeroline=False),
                yaxis=dict(title="Score", gridcolor=brd, zeroline=False),
                showlegend=False,
            )
            scatter_figs.append(f_scat)

        # Top 3 résumé
        top3_badges = html.Div([
            html.Div("🏆 Top 3 features prédictives du score", style={
                "color": sub, "fontSize": "10px", "textTransform": "uppercase",
                "letterSpacing": "2px", "marginBottom": "10px",
            }),
            html.Div([
                html.Div([
                    html.Span(f"#{i+1}  ", style={"color": _CLUSTER_COLORS_HEX[i], "fontFamily": t["font"],
                                                   "fontSize": "16px", "fontWeight": "700"}),
                    html.Span(_FEAT_LABELS.get(f, f), style={"color": txt, "fontSize": "13px"}),
                    html.Span(f"  r={co['spearman_r'][i]:+.3f}",
                              style={"color": t["accent1"], "fontSize": "11px",
                                     "fontFamily": t["font"], "marginLeft": "8px"}),
                ], style={"padding": "8px 12px", "borderLeft": f"3px solid {_CLUSTER_COLORS_HEX[i]}",
                          "marginBottom": "6px", "background": bg, "borderRadius": "0 6px 6px 0"})
                for i, f in enumerate(co["top3"])
            ]),
        ])

        tab_correlation = html.Div([
            html.Div([
                make_card([dcc.Graph(figure=fig_corr, config=_make_plotly_cfg())],
                           theme, {"flex": "2", "minWidth": "320px"}),
                make_card([top3_badges], theme, {"flex": "1", "minWidth": "240px"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"}),
            html.Div([
                make_card([dcc.Graph(figure=f, config=_make_plotly_cfg())],
                           theme, {"flex": "1", "minWidth": "280px"})
                for f in scatter_figs
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # ASSEMBLAGE FINAL
    # ══════════════════════════════════════════════════════════════════════════
    tab_style = {
        "background": card, "color": sub, "border": f"1px solid {brd}",
        "borderRadius": "6px 6px 0 0", "padding": "8px 18px",
        "fontSize": "12px", "fontFamily": t["font_body"], "cursor": "pointer",
    }
    tab_selected_style = {
        **tab_style, "color": t["accent1"],
        "borderBottom": f"2px solid {t['accent1']}",
        "fontWeight": "700",
    }

    return html.Div([
        html.Div([
            html.Div([
                html.Div("🧬 Analyse Comportementale · Shooter", style={
                    "color": t["accent1"], "fontSize": "22px",
                    "fontWeight": "700", "fontFamily": t["font"], "marginBottom": "4px",
                }),
                html.Div("Clustering · Progression intra-joueur · Corrélation features→score",
                         style={"color": sub, "fontSize": "13px"}),
            ], style={"flex": "1"}),
            data_badge(is_real, theme),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "flex-start", "marginBottom": "20px"}),

        kpi_strip,

        dcc.Tabs(id="profils-tabs", value=active_tab, children=[
            dcc.Tab(label="① Clustering Comportemental", value="clustering",
                    style=tab_style, selected_style=tab_selected_style,
                    children=[html.Div(tab_clustering, style={"paddingTop": "16px"})]),
            dcc.Tab(label="② Progression Intra-Joueur", value="progression",
                    style=tab_style, selected_style=tab_selected_style,
                    children=[html.Div(tab_progression, style={"paddingTop": "16px"})]),
            dcc.Tab(label="③ Corrélation → Score", value="correlation",
                    style=tab_style, selected_style=tab_selected_style,
                    children=[html.Div(tab_correlation, style={"paddingTop": "16px"})]),
        ], style={"marginBottom": "0"}),
    ])


def page_classifier(theme, df_real=None):
    t       = THEMES[theme]
    is_real = df_real is not None

    return html.Div([
        html.Div([
            html.Div([
                html.Div(" Classificateur", style={"color": t["accent1"], "fontSize": "22px",
                                                      "fontWeight": "700", "fontFamily": t["font"], "marginBottom": "4px"}),
                html.Div("Identification du profil d'un nouveau joueur en temps réel",
                         style={"color": t["subtext"], "fontSize": "13px"}),
            ]),
            data_badge(is_real, theme),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "flex-start", "marginBottom": "24px"}),

        html.Div([
            make_card([
                html.Div("Nouveau joueur", style={"color": t["subtext"], "fontSize": "11px",
                                                   "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "16px"}),
                html.Div("Nom", style={"color": t["subtext"], "fontSize": "11px", "marginBottom": "6px"}),
                dcc.Input(placeholder="ex: Nouveau joueur", style={
                    "background": t["bg"], "border": f"1px solid {t['border']}",
                    "color": t["text"], "padding": "8px 12px", "borderRadius": "6px",
                    "fontFamily": t["font_body"], "width": "100%", "marginBottom": "16px",
                }),
                *[html.Div([
                    html.Div(feat, style={"color": t["subtext"], "fontSize": "11px", "marginBottom": "4px"}),
                    dcc.Slider(0, 1, 0.01, value=round(np.random.uniform(0.3, 0.9), 2),
                               marks=None, tooltip={"placement": "bottom"}, className="custom-slider"),
                    html.Div(style={"marginBottom": "12px"}),
                ]) for feat in features_list],
                html.Button(" CLASSIFIER", style={
                    "background": t["gradient"], "border": "none", "color": "#000",
                    "padding": "12px 32px", "borderRadius": "6px", "cursor": "pointer",
                    "fontFamily": t["font"], "fontSize": "14px", "fontWeight": "700",
                    "letterSpacing": "2px", "width": "100%", "marginTop": "8px",
                }),
            ], theme, {"flex": "1"}),

            html.Div([
                make_card([
                    html.Div("Résultat", style={"color": t["subtext"], "fontSize": "11px",
                                                "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "16px"}),
                    html.Div([
                        html.Div("PROFIL IDENTIFIÉ", style={"color": t["subtext"], "fontSize": "11px", "letterSpacing": "2px"}),
                        html.Div(" PRÉCIS", style={"color": t["accent1"], "fontSize": "36px",
                                                      "fontFamily": t["font"], "fontWeight": "700", "marginTop": "8px"}),
                        html.Div("Confiance : 87%", style={"color": t["accent2"], "fontSize": "14px", "marginTop": "4px"}),
                    ], style={"textAlign": "center", "padding": "20px 0"}),
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Span(p, style={"color": c, "fontFamily": t["font_body"], "fontSize": "12px"}),
                                html.Span(f"{v}%", style={"color": t["subtext"], "fontSize": "12px"}),
                            ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
                            html.Div(html.Div(style={
                                "width": f"{v}%", "height": "6px", "background": c,
                                "borderRadius": "3px", "transition": "width 0.5s ease",
                            }), style={"background": t["border"], "borderRadius": "3px", "marginBottom": "10px"}),
                        ])
                        for p, v, c in [("Précis",87,"#69FF47"),("Prudent",8,"#00E5FF"),
                                        ("Agressif",3,"#FF4C6A"),("Chaotique",2,"#FFB800")]
                    ]),
                ], theme, {"marginBottom": "16px"}),
                make_card([
                    html.Div("Radar comparatif", style={"color": t["subtext"], "fontSize": "11px",
                                                         "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "8px"}),
                    dcc.Graph(figure=make_radar_fig("Précis", theme), config={"displayModeBar": False}),
                ], theme),
            ], style={"flex": "1", "display": "flex", "flexDirection": "column"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


def page_agent(theme, df_real=None):
    t       = THEMES[theme]
    is_real = df_real is not None

    return html.Div([
        html.Div([
            html.Div([
                html.Div(" Agent Imitateur", style={"color": t["accent1"], "fontSize": "22px",
                                                       "fontWeight": "700", "fontFamily": t["font"], "marginBottom": "4px"}),
                html.Div("Rejoue frame par frame les vrais inputs d'un joueur enregistré",
                         style={"color": t["subtext"], "fontSize": "13px"}),
            ]),
            data_badge(is_real, theme),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "alignItems": "flex-start", "marginBottom": "24px"}),

        html.Div([
            make_card([
                html.Div("Configuration de l'agent", style={"color": t["subtext"], "fontSize": "11px",
                                                              "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "16px"}),

                html.Div("Jeu à lancer", style={"color": t["subtext"], "fontSize": "11px", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="dropdown-agent-game",
                    options=[
                        {"label": " Reflex",    "value": "reflex"},
                        {"label": " Labyrinth", "value": "labyrinth"},
                        {"label": " Shooter",   "value": "shooter"},
                        {"label": " Racing",    "value": "racing"},
                    ],
                    value="reflex",
                    clearable=False,
                    style={"background": t["card"], "color": "#000", "borderRadius": "6px", "marginBottom": "16px"},
                ),

                html.Div("Joueur à imiter", style={"color": t["subtext"], "fontSize": "11px", "marginBottom": "6px"}),
                dcc.Dropdown(
                    id="dropdown-agent-player",
                    options=[{"label": p, "value": p} for p in mock_players],
                    placeholder="Sélectionner un joueur...",
                    style={"background": t["card"], "color": "#000", "borderRadius": "6px", "marginBottom": "16px"},
                ),

                html.Div("Fidélité d'imitation", style={"color": t["subtext"], "fontSize": "11px", "marginBottom": "6px"}),
                dcc.Slider(
                    id="slider-agent-noise",
                    min=0, max=100, step=1, value=80,
                    marks={0: "Libre", 50: "Mixte", 100: "Fidèle"},
                    tooltip={"placement": "bottom"},
                    className="custom-slider",
                ),
                html.Div(style={"height": "20px"}),

                html.Button("▶ LANCER L'AGENT", id="btn-launch-agent", n_clicks=0, style={
                    "background": t["gradient"], "border": "none", "color": "#000",
                    "padding": "12px 32px", "borderRadius": "6px", "cursor": "pointer",
                    "fontFamily": t["font"], "fontSize": "14px", "fontWeight": "700",
                    "letterSpacing": "2px", "width": "100%",
                }),
                html.Div(style={"height": "8px"}),
                html.Button("⏹ ARRÊTER", id="btn-stop-agent", n_clicks=0, style={
                    "background": "transparent", "border": f"1px solid {t['accent3']}",
                    "color": t["accent3"], "padding": "10px 32px", "borderRadius": "6px",
                    "cursor": "pointer", "fontFamily": t["font"], "fontSize": "13px", "width": "100%",
                }),
                html.Div(id="agent-feedback", style={"marginTop": "12px", "fontSize": "12px"}),
            ], theme, {"flex": "1"}),

            html.Div([
                make_card([
                    html.Div("Comparaison Humain vs Agent IA", style={"color": t["subtext"], "fontSize": "11px",
                                                                        "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "12px"}),
                    dcc.Graph(figure=make_agent_comparison(theme), config={"displayModeBar": False}),
                ], theme, {"marginBottom": "16px"}),
                make_card([
                    html.Div("Mode Replay", style={"color": t["subtext"], "fontSize": "11px",
                                                    "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "16px"}),
                    html.Div([
                        html.Div("Frame-by-Frame", style={"color": t["accent1"], "fontSize": "32px",
                                                           "fontFamily": t["font"], "fontWeight": "700", "textAlign": "center"}),
                        html.Div("Rejoue les inputs réels depuis Supabase",
                                 style={"color": t["subtext"], "fontSize": "13px", "textAlign": "center", "marginTop": "8px"}),
                        html.Div([
                            html.Div("Bruit gaussien σ", style={"color": t["subtext"], "fontSize": "11px", "marginBottom": "4px"}),
                            html.Div("configurable via le slider Fidélité",
                                     style={"color": t["accent2"], "fontSize": "12px", "fontFamily": t["font_body"]}),
                        ], style={"marginTop": "16px", "padding": "12px", "background": t["bg"],
                                  "borderRadius": "6px", "border": f"1px solid {t['border']}"}),
                    ]),
                ], theme),
            ], style={"flex": "2", "display": "flex", "flexDirection": "column"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


# ─────────────────────────────────────────────
# PAGE POST-SESSION
# ─────────────────────────────────────────────

def page_postsession(theme, player_name: str, game_id: str, summary_data=None):
    t = THEMES[theme]
    game_labels = {"reflex": " Reflex", "labyrinth": " Labyrinth",
                   "shooter": " Shooter", "racing": " Racing"}
    game_label = game_labels.get(game_id, game_id.upper())

    # Chargement stats brutes depuis Supabase
    session_data, all_game_data = None, []
    player_sessions_game = []
    try:
        from core.supabase_client import fetch_sessions_by_player, fetch_sessions_by_game
        player_sessions = fetch_sessions_by_player(player_name)
        all_game_data   = fetch_sessions_by_game(game_id)
        player_sessions_game = [s for s in player_sessions if s.get("game_id") == game_id]
        if player_sessions_game:
            session_data = player_sessions_game[-1]  # Dernière session
    except Exception:
        pass

    # Calculs classement
    if session_data and all_game_data:
        score = session_data.get("score", 0)
        scores_sorted  = sorted([s.get("score", 0) for s in all_game_data], reverse=True)
        rank_global    = next((i+1 for i, s in enumerate(scores_sorted) if s <= score), len(all_game_data))
        pct_global     = round((1 - rank_global / max(len(all_game_data), 1)) * 100, 1)
        personal_best  = max(s.get("score", 0) for s in player_sessions_game)
        global_best    = scores_sorted[0] if scores_sorted else 1
        bar_global_pct = f"{min(100, round(score / max(global_best, 1) * 100, 1))}%"
        bar_perso_pct  = f"{min(100, round(score / max(personal_best, 1) * 100, 1))}%"
    else:
        score = 0; rank_global = "?"; pct_global = "?"; personal_best = 0
        bar_global_pct = "0%"; bar_perso_pct = "0%"

    def stat_row(label, value, color=None):
        return html.Div([
            html.Span(label, style={"color": t["subtext"], "fontSize": "12px", "flex": "1"}),
            html.Span(str(value), style={"color": color or t["accent1"], "fontSize": "14px",
                                          "fontWeight": "700", "fontFamily": t["font"]}),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "padding": "8px 0", "borderBottom": f"1px solid {t['border']}"})

    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Div(f"Résumé de session — {player_name}",
                         style={"color": t["accent1"], "fontSize": "22px",
                                "fontWeight": "700", "fontFamily": t["font"], "marginBottom": "4px"}),
                html.Div(f"{game_label} · Analyse IA en cours...",
                         style={"color": t["subtext"], "fontSize": "13px"}),
            ]),
        ], style={"marginBottom": "24px"}),

        html.Div([
            # Colonne gauche : stats brutes + classement
            html.Div([
                make_card([
                    html.Div("Performances de la session", style={
                        "color": t["subtext"], "fontSize": "11px",
                        "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "12px"
                    }),
                    stat_row("Score", session_data.get("score", "—") if session_data else "—"),
                    stat_row("Durée", f"{session_data.get('duration_sec', 0):.0f}s" if session_data else "—"),
                    stat_row("Pression boutons", f"{session_data.get('btn_press_rate', 0):.3f}/s" if session_data else "—"),
                    stat_row("Agitation joystick lx", f"{session_data.get('lx_std', 0):.3f}" if session_data else "—"),
                    stat_row("Régularité", f"{session_data.get('input_regularity', 0):.2f}" if session_data else "—"),
                ], theme, {"marginBottom": "16px"}),

                make_card([
                    html.Div("Classements", style={
                        "color": t["subtext"], "fontSize": "11px",
                        "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "16px"
                    }),
                    html.Div(f"Rang global : {rank_global}/{len(all_game_data)} — {pct_global}% battus",
                             style={"color": t["text"], "fontSize": "13px", "marginBottom": "8px"}),
                    html.Div(html.Div(style={
                        "width": bar_global_pct, "height": "8px",
                        "background": t["gradient"], "borderRadius": "4px",
                    }), style={"background": t["border"], "borderRadius": "4px", "marginBottom": "16px"}),

                    html.Div(f"Meilleur perso : {personal_best}",
                             style={"color": t["text"], "fontSize": "13px", "marginBottom": "8px"}),
                    html.Div(html.Div(style={
                        "width": bar_perso_pct, "height": "8px",
                        "background": t["accent2"], "borderRadius": "4px",
                    }), style={"background": t["border"], "borderRadius": "4px"}),
                ], theme),
            ], style={"flex": "1"}),

            # Colonne droite : résumé LLM (polling)
            html.Div([
                make_card([
                    html.Div("Analyse IA", style={
                        "color": t["subtext"], "fontSize": "11px",
                        "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "16px"
                    }),
                    _render_summary_card(summary_data, t, compact=False) if summary_data else html.Div([
                        html.Div("Génération du résumé IA en cours...",
                                 style={"color": t["subtext"], "fontSize": "13px",
                                        "textAlign": "center", "marginBottom": "12px"}),
                        html.Div(style={
                            "width": "40px", "height": "40px", "margin": "0 auto",
                            "border": f"3px solid {t['border']}",
                            "borderTop": f"3px solid {t['accent1']}",
                            "borderRadius": "50%",
                            "animation": "spin 1s linear infinite",
                        }),
                    ], style={"padding": "32px", "textAlign": "center"}),
                ], theme),
            ], style={"flex": "2"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap"}),
    ])


# ─────────────────────────────────────────────
# PAGE CHATBOT
# ─────────────────────────────────────────────

def page_chat(theme):
    t = THEMES[theme]

    suggestions = [
        "Comment je me classe globalement ?",
        "Qui est le meilleur joueur sur shooter ?",
        "Analyse mon style de jeu",
        "Donne-moi des conseils pour progresser",
    ]

    return html.Div([
        # Header
        html.Div([
            html.Div(" Chat IA Gaming", style={
                "color": t["accent1"], "fontSize": "22px",
                "fontWeight": "700", "fontFamily": t["font"], "marginBottom": "4px"
            }),
            html.Div("Pose tes questions sur tes performances, classements et conseils",
                     style={"color": t["subtext"], "fontSize": "13px"}),
        ], style={"marginBottom": "24px"}),

        # Zone de chat
        make_card([
            # Messages scrollables
            html.Div(
                id="chat-messages-container",
                children=[
                    html.Div([
                        html.Span("🤖", style={"fontSize": "18px", "marginRight": "10px"}),
                        html.Span("Bonjour ! Je suis ton coach IA. Pose-moi une question sur tes performances.",
                                  style={"color": t["text"], "fontSize": "13px"}),
                    ], style={
                        "background": t["bg"], "borderRadius": "8px", "padding": "12px 16px",
                        "marginBottom": "8px", "borderLeft": f"3px solid {t['accent1']}",
                    }),
                ],
                style={"minHeight": "350px", "maxHeight": "420px", "overflowY": "auto",
                       "marginBottom": "16px", "padding": "4px"},
            ),

            # Suggestions rapides
            html.Div([
                html.Button(s, id={"type": "chat-suggestion", "index": i}, n_clicks=0, style={
                    "background": "transparent", "border": f"1px solid {t['border']}",
                    "color": t["subtext"], "borderRadius": "16px", "padding": "4px 12px",
                    "fontSize": "11px", "cursor": "pointer", "marginRight": "6px",
                    "fontFamily": t["font_body"],
                }) for i, s in enumerate(suggestions)
            ], style={"marginBottom": "12px", "flexWrap": "wrap", "display": "flex", "gap": "4px"}),

            # Input + bouton
            html.Div([
                dcc.Input(
                    id="chat-input",
                    type="text",
                    placeholder="Pose ta question...",
                    debounce=False,
                    n_submit=0,
                    style={
                        "flex": "1", "background": t["bg"],
                        "border": f"1px solid {t['border']}", "borderRadius": "6px",
                        "color": t["text"], "padding": "10px 14px", "fontSize": "13px",
                        "fontFamily": t["font_body"], "outline": "none",
                    },
                ),
                html.Button("▶ Envoyer", id="btn-chat-send", n_clicks=0, style={
                    "background": t["gradient"], "border": "none", "color": "#000",
                    "padding": "10px 24px", "borderRadius": "6px", "cursor": "pointer",
                    "fontFamily": t["font"], "fontSize": "13px", "fontWeight": "700",
                    "marginLeft": "8px", "whiteSpace": "nowrap",
                }),
            ], style={"display": "flex", "alignItems": "center"}),

            html.Div(id="chat-loading", style={"color": t["subtext"], "fontSize": "11px",
                                                "marginTop": "8px", "minHeight": "16px"}),
        ], theme),
    ])


# ─────────────────────────────────────────────
# PAGE LEADERBOARD
# ─────────────────────────────────────────────

_GAME_LABELS = {
    "reflex": " Reflex", "labyrinth": " Labyrinth",
    "shooter": " Shooter", "racing": " Racing",
}
_GAME_OPTIONS = [{"label": "Tous les jeux", "value": "all"}] + [
    {"label": v, "value": k} for k, v in _GAME_LABELS.items()
]
_MEDALS = ["🥇", "🥈", "🥉"]
_PODIUM_COLORS = {
    0: ("linear-gradient(180deg,#FFD700 0%,#B8860B 100%)", "#FFD700"),
    1: ("linear-gradient(180deg,#C0C0C0 0%,#808080 100%)", "#C0C0C0"),
    2: ("linear-gradient(180deg,#CD7F32 0%,#8B4513 100%)", "#CD7F32"),
}
_PODIUM_HEIGHTS = {0: "160px", 1: "120px", 2: "90px"}


def _make_podium_block(rank, player, avg_score, n_sessions, t):
    grad, col = _PODIUM_COLORS[rank]
    height = _PODIUM_HEIGHTS[rank]
    emoji = _MEDALS[rank]
    return html.Div([
        # Avatar + name above the block
        html.Div([
            html.Div(emoji, style={"fontSize": "28px", "marginBottom": "4px"}),
            html.Div(player, style={
                "color": t["text"], "fontFamily": t["font"],
                "fontWeight": "700", "fontSize": "13px",
                "textAlign": "center", "wordBreak": "break-word",
                "textShadow": f"0 0 8px {col}",
            }),
            html.Div(f"{int(avg_score)} pts moy.", style={
                "color": col, "fontSize": "11px", "marginTop": "4px",
                "textAlign": "center",
            }),
            html.Div(f"{n_sessions} session{'s' if n_sessions > 1 else ''}", style={
                "color": t["subtext"], "fontSize": "10px", "textAlign": "center",
            }),
        ], style={"marginBottom": "8px"}),
        # Podium block itself
        html.Div(
            html.Div(str(rank + 1), style={
                "color": "#000", "fontFamily": t["font"], "fontWeight": "900",
                "fontSize": "28px", "textAlign": "center", "lineHeight": height,
            }),
            style={
                "background": grad, "height": height, "width": "80px",
                "borderRadius": "8px 8px 0 0", "boxShadow": f"0 0 20px {col}55",
            }
        ),
    ], style={
        "display": "flex", "flexDirection": "column",
        "alignItems": "center", "justifyContent": "flex-end",
    })


def _build_leaderboard_content(df, game_filter, player_search, t, theme, table_player_filter=None):
    """Construit le contenu filtré du leaderboard (podium + table + graphiques)."""
    # ── Filtrage global (podium + graphiques) ─────────────────────────────────
    dff = df.copy()
    if game_filter and game_filter != "all":
        dff = dff[dff["game_id"] == game_filter]
    if player_search and player_search.strip():
        mask = dff["player_name"].str.contains(player_search.strip(), case=False, na=False)
        dff = dff[mask]

    if dff.empty:
        return html.Div("Aucune session ne correspond aux filtres.",
                        style={"color": t["subtext"], "textAlign": "center", "padding": "40px"})

    # ── Stats joueurs ─────────────────────────────────────────────────────────
    player_stats = (
        dff.groupby("player_name")
        .agg(avg_score=("score", "mean"), max_score=("score", "max"),
             n_sessions=("score", "count"))
        .sort_values("avg_score", ascending=False)
        .reset_index()
    )
    top3 = player_stats.head(3)
    top5_sessions = dff.nlargest(5, "score").reset_index(drop=True)

    # ── Podium ────────────────────────────────────────────────────────────────
    podium_order = [1, 0, 2]  # Silver left, Gold center, Bronze right
    podium_blocks = []
    for display_pos, rank in enumerate(podium_order):
        if rank < len(top3):
            row = top3.iloc[rank]
            podium_blocks.append(
                _make_podium_block(rank, row["player_name"], row["avg_score"], row["n_sessions"], t)
            )

    podium_section = html.Div([
        html.Div(" Podium des Meilleurs Joueurs", style={
            "color": t["subtext"], "fontSize": "11px", "textTransform": "uppercase",
            "letterSpacing": "2px", "marginBottom": "24px",
        }),
        html.Div(podium_blocks, style={
            "display": "flex", "justifyContent": "center",
            "alignItems": "flex-end", "gap": "24px", "marginBottom": "16px",
        }),
        # Podium base
        html.Div(style={
            "height": "6px", "background": t["gradient"],
            "borderRadius": "3px", "boxShadow": t["glow"],
        }),
    ], style={"marginBottom": "8px"})

    # ── Top 5 sessions ────────────────────────────────────────────────────────
    rank_colors = {0: "#FFD700", 1: "#C0C0C0", 2: "#CD7F32"}
    table_rows = []
    for i, row in top5_sessions.iterrows():
        game_col = {
            "reflex": t["accent1"], "labyrinth": t["accent2"],
            "shooter": t["accent3"], "racing": "#FFB800",
        }.get(row.get("game_id", ""), t["subtext"])
        table_rows.append(html.Tr([
            html.Td(
                html.Div([
                    html.Span(_MEDALS[i] if i < 3 else f"#{i+1}",
                              style={"fontSize": "16px" if i < 3 else "12px"}),
                ], style={"textAlign": "center"}),
                style={"padding": "10px 8px", "borderBottom": f"1px solid {t['border']}"}
            ),
            html.Td(row.get("player_name", "—"),
                    style={"color": rank_colors.get(i, t["text"]) if i < 3 else t["text"],
                           "fontWeight": "700" if i < 3 else "400",
                           "fontFamily": t["font"] if i < 3 else t["font_body"],
                           "padding": "10px 8px", "borderBottom": f"1px solid {t['border']}",
                           "textShadow": f"0 0 8px {rank_colors[i]}" if i < 3 else "none"}),
            html.Td(
                html.Span(_GAME_LABELS.get(row.get("game_id", ""), row.get("game_id", "—")),
                          style={"color": game_col, "fontSize": "12px"}),
                style={"padding": "10px 8px", "borderBottom": f"1px solid {t['border']}"}),
            html.Td(
                html.Div([
                    html.Span(f"{int(row.get('score', 0))}",
                              style={"color": t["accent1"], "fontFamily": t["font"],
                                     "fontWeight": "700", "fontSize": "15px"}),
                    html.Span(" pts", style={"color": t["subtext"], "fontSize": "10px"}),
                ]),
                style={"padding": "10px 8px", "borderBottom": f"1px solid {t['border']}"}),
            html.Td(f"{int(row.get('duration_sec', 0))}s",
                    style={"color": t["subtext"], "fontSize": "12px",
                           "padding": "10px 8px", "borderBottom": f"1px solid {t['border']}"}),
        ]))

    top5_table = html.Table([
        html.Thead(html.Tr([
            html.Th(col, style={
                "color": t["subtext"], "fontSize": "10px", "padding": "6px 8px",
                "textTransform": "uppercase", "letterSpacing": "1px",
                "borderBottom": f"1px solid {t['border']}", "textAlign": "left",
            }) for col in ["#", "Joueur", "Jeu", "Score", "Durée"]
        ])),
        html.Tbody(table_rows),
    ], style={"width": "100%", "borderCollapse": "collapse"})

    # ── Graphiques Plotly ─────────────────────────────────────────────────────
    bg   = t["bg"]
    txt  = t["text"]
    sub  = t["subtext"]
    brd  = t["border"]
    card = t["card"]

    plot_cfg = dict(
        paper_bgcolor=card, plot_bgcolor=card,
        font=dict(color=txt, family=t["font_body"], size=11),
        margin=dict(l=10, r=10, t=30, b=10),
    )

    # Graphique 1 : Top 10 joueurs (bar horizontal)
    top10 = player_stats.head(10)
    bar_colors = [_PODIUM_COLORS[i][1] if i < 3 else t["accent1"] for i in range(len(top10))]
    fig_players = go.Figure(go.Bar(
        x=top10["avg_score"].round(1).tolist(),
        y=top10["player_name"].tolist(),
        orientation="h",
        marker=dict(
            color=bar_colors,
            line=dict(width=0),
        ),
        text=[f"{v:.0f}" for v in top10["avg_score"]],
        textposition="inside",
        textfont=dict(color="#000", size=10, family=t["font"]),
        hovertemplate="<b>%{y}</b><br>Moy. score : %{x:.0f}<extra></extra>",
    ))
    fig_players.update_layout(
        **plot_cfg,
        xaxis=dict(gridcolor=brd, zeroline=False, showgrid=True),
        yaxis=dict(gridcolor=brd, autorange="reversed", categoryorder="total ascending"),
        showlegend=False,
        height=280,
    )

    # Graphique 2 : Score moyen par jeu (bar verticale)
    game_avg = (dff.groupby("game_id")["score"].agg(["mean", "max", "count"])
                .reset_index().sort_values("mean", ascending=False))
    game_labels_chart = [_GAME_LABELS.get(g, g) for g in game_avg["game_id"]]
    game_colors_chart = [
        {"reflex": t["accent1"], "labyrinth": t["accent2"],
         "shooter": t["accent3"], "racing": "#FFB800"}.get(g, t["subtext"])
        for g in game_avg["game_id"]
    ]
    fig_games = go.Figure()
    fig_games.add_trace(go.Bar(
        x=game_labels_chart, y=game_avg["mean"].round(1).tolist(),
        name="Score moyen",
        marker=dict(color=game_colors_chart, opacity=0.9, line=dict(width=0)),
        text=[f"{v:.0f}" for v in game_avg["mean"]],
        textposition="outside",
        textfont=dict(size=10, family=t["font"]),
        hovertemplate="<b>%{x}</b><br>Score moy : %{y:.0f}<extra></extra>",
    ))
    fig_games.add_trace(go.Scatter(
        x=game_labels_chart, y=game_avg["max"].tolist(),
        name="Record", mode="markers",
        marker=dict(symbol="star", size=14, color=t["accent2"],
                    line=dict(color=t["bg"], width=1)),
        hovertemplate="<b>%{x}</b><br>Record : %{y}<extra></extra>",
    ))
    fig_games.update_layout(
        **plot_cfg,
        xaxis=dict(gridcolor=brd),
        yaxis=dict(gridcolor=brd, zeroline=False),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=10)),
        barmode="group", height=280,
    )

    # ── Tableau complet des sessions (filtre joueur indépendant) ─────────────
    dff_table = dff.copy()
    if table_player_filter and table_player_filter != "all":
        dff_table = dff_table[dff_table["player_name"] == table_player_filter]
    all_sessions_sorted = dff_table.sort_values("score", ascending=False).reset_index(drop=True)

    _game_col_map = {
        "reflex": t["accent1"], "labyrinth": t["accent2"],
        "shooter": t["accent3"], "racing": "#FFB800",
    }

    def _score_bar(score, max_score):
        pct = min(100, round(score / max(max_score, 1) * 100))
        return html.Div(
            html.Div(style={
                "width": f"{pct}%", "height": "4px",
                "background": t["gradient"], "borderRadius": "2px",
                "boxShadow": t["glow"] if pct >= 80 else "none",
            }),
            style={"background": t["border"], "borderRadius": "2px",
                   "width": "80px", "display": "inline-block", "verticalAlign": "middle"},
        )

    max_score_all = int(dff_table["score"].max()) if not dff_table.empty else 1
    all_cols = ["#", "Joueur", "Jeu", "Score", "Durée"]
    has_regularity = "input_regularity" in dff.columns
    has_btn_rate   = "btn_press_rate" in dff.columns
    has_date       = "created_at" in dff.columns
    if has_btn_rate:
        all_cols.append("Boutons/s")
    if has_regularity:
        all_cols.append("Régularité")
    if has_date:
        all_cols.append("Date")

    all_rows = []
    for i, row in all_sessions_sorted.iterrows():
        rank_num = i + 1
        rank_cell = (
            html.Span(_MEDALS[i], style={"fontSize": "14px"})
            if i < 3
            else html.Span(f"#{rank_num}", style={"color": t["subtext"], "fontSize": "11px"})
        )
        gcol = _game_col_map.get(row.get("game_id", ""), t["subtext"])
        score_val = int(row.get("score", 0))

        cells = [
            html.Td(rank_cell,
                    style={"padding": "8px 8px", "borderBottom": f"1px solid {t['border']}",
                           "textAlign": "center", "width": "40px"}),
            html.Td(row.get("player_name", "—"),
                    style={"padding": "8px 8px", "borderBottom": f"1px solid {t['border']}",
                           "color": t["text"], "fontWeight": "600" if i < 3 else "400",
                           "fontFamily": t["font"] if i < 3 else t["font_body"],
                           "fontSize": "13px"}),
            html.Td(
                html.Span(_GAME_LABELS.get(row.get("game_id", ""), row.get("game_id", "—")),
                          style={"color": gcol, "fontSize": "11px",
                                 "border": f"1px solid {gcol}44",
                                 "borderRadius": "4px", "padding": "1px 6px"}),
                style={"padding": "8px 8px", "borderBottom": f"1px solid {t['border']}"}),
            html.Td([
                html.Span(f"{score_val}",
                          style={"color": t["accent1"], "fontFamily": t["font"],
                                 "fontWeight": "700", "fontSize": "14px", "marginRight": "6px"}),
                _score_bar(score_val, max_score_all),
            ], style={"padding": "8px 8px", "borderBottom": f"1px solid {t['border']}",
                      "whiteSpace": "nowrap"}),
            html.Td(f"{int(row.get('duration_sec', 0))}s",
                    style={"padding": "8px 8px", "borderBottom": f"1px solid {t['border']}",
                           "color": t["subtext"], "fontSize": "12px"}),
        ]
        if has_btn_rate:
            cells.append(html.Td(
                f"{row.get('btn_press_rate', 0):.2f}",
                style={"padding": "8px 8px", "borderBottom": f"1px solid {t['border']}",
                       "color": t["subtext"], "fontSize": "12px", "textAlign": "right"}))
        if has_regularity:
            reg = row.get("input_regularity", 0)
            reg_col = t["accent2"] if reg >= 0.7 else (t["accent3"] if reg >= 0.4 else t["subtext"])
            cells.append(html.Td(
                f"{reg:.2f}",
                style={"padding": "8px 8px", "borderBottom": f"1px solid {t['border']}",
                       "color": reg_col, "fontSize": "12px", "textAlign": "right"}))
        if has_date:
            raw_date = str(row.get("created_at", ""))[:10]
            cells.append(html.Td(
                raw_date,
                style={"padding": "8px 8px", "borderBottom": f"1px solid {t['border']}",
                       "color": t["subtext"], "fontSize": "11px"}))

        row_style = {"transition": "background 0.15s"}
        all_rows.append(html.Tr(cells, style=row_style))

    sessions_table = html.Div([
        html.Table([
            html.Thead(html.Tr([
                html.Th(col, style={
                    "color": t["subtext"], "fontSize": "10px", "padding": "8px 8px",
                    "textTransform": "uppercase", "letterSpacing": "1px",
                    "borderBottom": f"2px solid {t['border']}",
                    "textAlign": "left", "position": "sticky", "top": "0",
                    "background": t["card"], "zIndex": "1",
                }) for col in all_cols
            ])),
            html.Tbody(all_rows),
        ], style={"width": "100%", "borderCollapse": "collapse"}),
    ], style={"maxHeight": "420px", "overflowY": "auto", "overflowX": "auto"})

    return html.Div([
        # Podium
        make_card(podium_section, theme, {"marginBottom": "16px"}),

        # Top 5 sessions + graphiques
        html.Div([
            make_card([
                html.Div(" Top 5 Sessions", style={
                    "color": t["subtext"], "fontSize": "11px",
                    "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "12px",
                }),
                top5_table,
            ], theme, {"flex": "1", "minWidth": "280px"}),

            html.Div([
                make_card([
                    html.Div("Top 10 Joueurs · Score Moyen", style={
                        "color": t["subtext"], "fontSize": "11px",
                        "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "8px",
                    }),
                    dcc.Graph(figure=fig_players, config={"displayModeBar": False}),
                ], theme, {"marginBottom": "16px"}),
                make_card([
                    html.Div("Score Moyen & Record par Jeu", style={
                        "color": t["subtext"], "fontSize": "11px",
                        "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "8px",
                    }),
                    dcc.Graph(figure=fig_games, config={"displayModeBar": False}),
                ], theme),
            ], style={"flex": "2", "display": "flex", "flexDirection": "column", "minWidth": "300px"}),
        ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"}),

        # Tableau complet des sessions
        make_card([
            html.Div([
                html.Div(" Toutes les Sessions", style={
                    "color": t["subtext"], "fontSize": "11px",
                    "textTransform": "uppercase", "letterSpacing": "2px",
                }),
                html.Div(f"{len(dff_table)} session{'s' if len(dff_table) > 1 else ''} · triées par score",
                         style={"color": t["subtext"], "fontSize": "10px"}),
            ], style={"display": "flex", "justifyContent": "space-between",
                      "alignItems": "center", "marginBottom": "12px"}),
            sessions_table,
        ], theme),
    ])


def page_leaderboard(theme, df_real):
    t = THEMES[theme]

    # Données : réelles ou mock
    if df_real is not None and len(df_real) >= 3:
        df = df_real.copy()
        is_real = True
    else:
        rng = np.random.default_rng(0)
        mock_gs = ["shooter", "reflex", "labyrinth", "racing"]
        df = pd.DataFrame({
            "player_name": [mock_players[i % len(mock_players)] for i in range(40)],
            "game_id":     [mock_gs[i % 4] for i in range(40)],
            "score":       rng.integers(50, 950, 40).tolist(),
            "duration_sec": rng.integers(30, 300, 40).tolist(),
        })
        is_real = False

    # ── KPIs globaux ──────────────────────────────────────────────────────────
    total_sessions = len(df)
    unique_players = df["player_name"].nunique()
    best_score     = int(df["score"].max()) if not df.empty else 0
    avg_score      = int(df["score"].mean()) if not df.empty else 0
    top_game_val   = df["game_id"].value_counts()
    top_game       = _GAME_LABELS.get(top_game_val.index[0], top_game_val.index[0]) if not top_game_val.empty else "—"
    top_player_val = df["player_name"].value_counts()
    top_player     = top_player_val.index[0] if not top_player_val.empty else "—"

    kpi_style = {
        "display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "20px",
    }

    def kpi(icon, label, value, color=None):
        c = color or t["accent1"]
        return html.Div([
            html.Div(icon, style={"fontSize": "20px", "marginBottom": "4px"}),
            html.Div(str(value), style={
                "color": c, "fontSize": "26px", "fontWeight": "700",
                "fontFamily": t["font"], "lineHeight": "1.1",
                "textShadow": f"0 0 10px {c}88",
            }),
            html.Div(label, style={
                "color": t["subtext"], "fontSize": "10px",
                "textTransform": "uppercase", "letterSpacing": "1.5px", "marginTop": "2px",
            }),
        ], style={
            "background": t["card"], "border": f"1px solid {t['border']}",
            "borderRadius": "10px", "padding": "14px 18px",
            "flex": "1", "minWidth": "110px", "maxWidth": "200px",
            "boxShadow": t["glow"],
        })

    kpi_strip = html.Div([
        kpi("📊", "Sessions",      total_sessions, t["accent1"]),
        kpi("👥", "Joueurs",       unique_players, t["accent2"]),
        kpi("🏆", "Record Absolu", best_score,     "#FFD700"),
        kpi("⭐", "Score Moyen",   avg_score,      t["accent3"]),
        kpi("🎮", "Jeu N°1",       top_game,       t["accent2"]),
        kpi("🔥", "Joueur Actif",  top_player,     t["accent1"]),
    ], style=kpi_style)

    return html.Div([
        # Header
        html.Div([
            html.Div([
                html.Div("🏆 Classement SISE Gaming", style={
                    "color": t["accent1"], "fontSize": "24px",
                    "fontWeight": "700", "fontFamily": t["font"],
                    "textShadow": t["glow"], "marginBottom": "4px",
                }),
                html.Div("Hall of Fame · Podium · Records · Statistiques",
                         style={"color": t["subtext"], "fontSize": "13px"}),
            ], style={"flex": "1"}),
            data_badge(is_real, theme),
        ], style={"display": "flex", "alignItems": "center",
                  "justifyContent": "space-between", "marginBottom": "20px"}),

        # KPI strip
        kpi_strip,

        # Barre de filtres globaux (podium + graphiques)
        make_card([
            html.Div("Filtres globaux", style={
                "color": t["subtext"], "fontSize": "10px",
                "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "10px",
            }),
            html.Div([
                dcc.Dropdown(
                    id="leaderboard-game-filter",
                    options=_GAME_OPTIONS,
                    value="all",
                    clearable=False,
                    placeholder="Tous les jeux",
                    style={
                        "background": t["bg"], "color": t["text"],
                        "border": f"1px solid {t['border']}",
                        "borderRadius": "6px", "flex": "1", "minWidth": "160px",
                        "fontFamily": t["font_body"], "fontSize": "12px",
                    },
                ),
                dcc.Input(
                    id="leaderboard-player-search",
                    type="text",
                    placeholder=" Rechercher un joueur...",
                    debounce=True,
                    style={
                        "flex": "2", "background": t["bg"],
                        "border": f"1px solid {t['border']}", "borderRadius": "6px",
                        "color": t["text"], "padding": "8px 12px",
                        "fontSize": "12px", "fontFamily": t["font_body"],
                        "outline": "none", "minWidth": "200px",
                    },
                ),
                html.Div(f"{'🟢 LIVE' if is_real else '🟡 MOCK'} · {total_sessions} sessions · {unique_players} joueurs",
                         style={"color": t["subtext"], "fontSize": "11px",
                                "alignSelf": "center", "whiteSpace": "nowrap"}),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "center"}),
        ], theme, {"marginBottom": "10px", "padding": "14px 20px"}),

        # Filtre joueur pour le tableau des sessions
        make_card([
            html.Div("Filtre Tableau — Joueur", style={
                "color": t["subtext"], "fontSize": "10px",
                "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "10px",
            }),
            dcc.Dropdown(
                id="leaderboard-player-dropdown",
                options=[{"label": "Tous les joueurs", "value": "all"}] + [
                    {"label": p, "value": p}
                    for p in sorted(df["player_name"].dropna().unique())
                ],
                value="all",
                clearable=False,
                placeholder="Sélectionner un joueur...",
                style={
                    "background": t["bg"], "color": t["text"],
                    "border": f"1px solid {t['border']}",
                    "borderRadius": "6px", "fontFamily": t["font_body"], "fontSize": "12px",
                },
            ),
        ], theme, {"marginBottom": "16px", "padding": "14px 20px"}),

        # Contenu dynamique (podium + table + graphiques)
        html.Div(
            id="leaderboard-dynamic-content",
            children=_build_leaderboard_content(df, "all", "", t, theme, None),
        ),
    ])


# ─────────────────────────────────────────────
# LAYOUT
# ─────────────────────────────────────────────
app.layout = html.Div([
    dcc.Store(id="theme-store",       data="cyberpunk"),
    dcc.Store(id="page-store",        data="game"),
    dcc.Store(id="sessions-store",    data=[]),
    dcc.Store(id="stats-store",       data={}),
    dcc.Store(id="summary-store",     data=[]),
    dcc.Store(id="agent-pid-store",   data=None),
    dcc.Store(id="profils-tab-store", data="clustering"),
    dcc.Store(id="url-params-store",        data={}),
    dcc.Store(id="chat-store",              data=[]),
    dcc.Store(id="postsession-summary-data",data=None),
    dcc.Location(id="url", refresh=False),
    dcc.Interval(id="refresh-interval",          interval=5000,  n_intervals=0),
    dcc.Interval(id="summary-refresh-interval",  interval=8000,  n_intervals=0),
    dcc.Interval(id="postsession-interval",      interval=2000,  n_intervals=0, disabled=True),

    html.Link(rel="stylesheet", href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Share+Tech+Mono&family=Exo+2:wght@400;700&family=IBM+Plex+Mono&family=VT323&family=Courier+Prime&family=Syne:wght@400;700;800&family=Space+Mono&display=swap"),

    html.Div(id="main-container", children=[
        # ── SIDEBAR ──
        html.Div(id="sidebar", children=[
            html.Div([
                html.Img(
                    src="/assets/logo_sise_gaming.png",
                    style={"width": "100%", "maxWidth": "270px", "objectFit": "contain",
                           "display": "block", "margin": "0 auto 28px auto",
                           "filter": "drop-shadow(0 0 14px rgba(199,36,177,0.7))",
                           "borderRadius": "12px"}
                ),
            ]),
            html.Div("NAVIGATION", style={"fontSize": "9px", "letterSpacing": "3px",
                                           "opacity": "0.4", "marginBottom": "12px", "padding": "0 4px"}),
            html.Div([
                html.Button([html.Span("🎮", style={"marginRight": "10px"}), "Live Game"],
                            id="nav-game",        n_clicks=0, className="nav-btn"),
                html.Button([html.Span("", style={"marginRight": "10px"}), "Résumés"],
                            id="nav-summary",     n_clicks=0, className="nav-btn"),
                html.Button([html.Span("", style={"marginRight": "10px"}), "Dashboard"],
                            id="nav-leaderboard", n_clicks=0, className="nav-btn"),
                html.Button([html.Span("", style={"marginRight": "10px"}), "Players Profils"],
                            id="nav-profils",     n_clicks=0, className="nav-btn"),
                html.Button([html.Span("", style={"marginRight": "10px"}), "Gaming Chatbot"],
                            id="nav-chat",        n_clicks=0, className="nav-btn"),
                html.Button([html.Span("", style={"marginRight": "10px"}), "IA Gamer"],
                            id="nav-agent",       n_clicks=0, className="nav-btn"),
                # nav-classifier conservé invisible pour la compatibilité des callbacks
                html.Button(id="nav-classifier", n_clicks=0,
                            style={"display": "none"}),
            ], style={"display": "flex", "flexDirection": "column", "gap": "4px", "marginBottom": "32px"}),
            html.Div("THÈME", style={"fontSize": "9px", "letterSpacing": "3px",
                                      "opacity": "0.4", "marginBottom": "12px", "padding": "0 4px"}),
            html.Div([
                html.Button(THEMES[th]["name"], id=f"theme-{th}", n_clicks=0,
                            className="theme-btn", **{"data-theme": th})
                for th in THEMES
            ], style={"display": "flex", "flexDirection": "column", "gap": "4px"}),
            html.Div([
                html.Div("Master SISE 2025–2026", style={"fontSize": "10px", "opacity": "0.4"}),
                html.Div("Projet IA Temps réel",  style={"fontSize": "10px", "opacity": "0.4"}),
            ], style={"position": "absolute", "bottom": "24px", "left": "24px"}),
        ]),
        html.Div(id="page-content", style={"flex": "1", "padding": "32px", "overflowY": "auto"}),
    ]),
], id="root")


# ─────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────
@app.callback(
    Output("sessions-store", "data"),
    Output("stats-store",    "data"),
    Input("refresh-interval", "n_intervals"),
)
def refresh_sessions(n):
    try:
        data  = fetch_latest_sessions(limit=200)
        data  = data if data else []
        stats = {
            "n_sessions": len(data),
            "n_players":  len(set(d["player_name"] for d in data)) if data else 0,
            "avg_score":  int(sum(d["score"] for d in data) / len(data)) if data else 0,
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
    [Input("nav-game","n_clicks"),    Input("nav-profils","n_clicks"),
     Input("nav-classifier","n_clicks"), Input("nav-agent","n_clicks"),
     Input("nav-summary","n_clicks"), Input("nav-chat","n_clicks"),
     Input("nav-leaderboard","n_clicks")],
    prevent_initial_call=True,
)
def update_page(g, p, c, a, s, ch, lb):
    ctx = dash.callback_context
    if not ctx.triggered:
        return "game"
    btn = ctx.triggered[0]["prop_id"].split(".")[0]
    return {"nav-game":"game","nav-profils":"profils",
            "nav-classifier":"classifier","nav-agent":"agent",
            "nav-summary":"summary","nav-chat":"chat",
            "nav-leaderboard":"leaderboard"}.get(btn, "game")


@app.callback(
    Output("leaderboard-dynamic-content", "children"),
    Input("leaderboard-game-filter",      "value"),
    Input("leaderboard-player-search",    "value"),
    Input("leaderboard-player-dropdown",  "value"),
    dash.dependencies.State("sessions-store", "data"),
    dash.dependencies.State("theme-store",    "data"),
    prevent_initial_call=True,
)
def update_leaderboard_filtered(game_filter, player_search, player_dropdown, sessions_data, theme):
    t = THEMES[theme]
    if sessions_data and len(sessions_data) >= 3:
        df = pd.DataFrame(sessions_data)
    else:
        rng = np.random.default_rng(0)
        mock_gs = ["shooter", "reflex", "labyrinth", "racing"]
        df = pd.DataFrame({
            "player_name": [mock_players[i % len(mock_players)] for i in range(40)],
            "game_id":     [mock_gs[i % 4] for i in range(40)],
            "score":       rng.integers(50, 950, 40).tolist(),
            "duration_sec": rng.integers(30, 300, 40).tolist(),
        })
    return _build_leaderboard_content(
        df,
        game_filter or "all",
        player_search or "",
        t, theme,
        table_player_filter=player_dropdown,
    )


@app.callback(
    Output("profils-tab-store", "data"),
    Input("profils-tabs", "value"),
    prevent_initial_call=True,
)
def save_profils_tab(tab):
    return tab


@app.callback(
    Output("main-container", "style"),
    Output("sidebar",        "style"),
    Output("page-content",   "children"),
    Input("theme-store",              "data"),
    Input("page-store",               "data"),
    Input("postsession-summary-data", "data"),
    Input("sessions-store",           "data"),
    dash.dependencies.State("url-params-store",  "data"),
    dash.dependencies.State("profils-tab-store", "data"),
)
def render_all(theme, page, summary_data, sessions_data, url_params, profils_tab):
    t       = THEMES[theme]
    df_real = pd.DataFrame(sessions_data) if sessions_data and len(sessions_data) >= 3 else None
    params  = url_params or {}

    container_style = {"display": "flex", "minHeight": "100vh",
                       "background": t["bg"], "color": t["text"], "fontFamily": t["font_body"]}
    sidebar_style   = {"width": "220px", "minHeight": "100vh",
                       "background": t["sidebar"], "borderRight": f"1px solid {t['border']}",
                       "padding": "24px", "position": "relative", "fontFamily": t["font_body"]}

    # Rendu lazy : seule la page active est calculée (+ postsession toujours légère)
    _page_builders = {
        "game":        lambda: page_game(theme, df_real),
        "profils":     lambda: page_profils(theme, df_real, active_tab=profils_tab or "clustering"),
        "classifier":  lambda: page_classifier(theme, df_real),
        "agent":       lambda: page_agent(theme, df_real),
        "summary":     lambda: page_summary(theme),
        "chat":        lambda: page_chat(theme),
        "postsession": lambda: page_postsession(theme, params.get("player", ""), params.get("game", ""), summary_data),
        "leaderboard": lambda: page_leaderboard(theme, df_real),
    }
    active_page = _page_builders.get(page, _page_builders["game"])()
    return container_style, sidebar_style, active_page


@app.callback(
    Output("stat-sessions",      "children"),
    Output("stat-players",       "children"),
    Output("stat-score",         "children"),
    Output("stat-reaction",      "children"),
    Output("data-badge-container","children"),
    Input("stats-store",  "data"),
    Input("theme-store",  "data"),
)
def update_stats(stats, theme):
    n_sess    = str(stats.get("n_sessions", 0)) if stats else "—"
    n_players = str(stats.get("n_players",  0)) if stats else "—"
    avg_score = str(stats.get("avg_score",  0)) if stats else "—"
    is_real   = bool(stats and stats.get("n_sessions", 0) > 0)
    return (
        stat_card("Sessions",      n_sess,    "▲ Supabase live" if is_real else "En attente…", theme).children,
        stat_card("Joueurs",       n_players, "Uniques", theme).children,
        stat_card("Score moyen",   avg_score, "Toutes sessions", theme).children,
        stat_card("Réaction moy.", "— ms",    "Temps de réponse", theme).children,
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
    dash.dependencies.State("input-player-name",   "value"),
    dash.dependencies.State("dropdown-game-select","value"),
    prevent_initial_call=True,
)
def launch_game(n_clicks, player_name, game_id):
    if not player_name or not player_name.strip():
        return "⚠️ Entre un nom de joueur avant de lancer.", \
               {"marginTop": "12px", "fontSize": "12px", "color": "#FFB800"}
    if not game_id:
        return "⚠️ Sélectionne un jeu.", \
               {"marginTop": "12px", "fontSize": "12px", "color": "#FFB800"}
    try:
        subprocess.Popen(
            [sys.executable, "main.py", game_id, player_name.strip()],
            cwd=ROOT_DIR,
        )
        return f"✅ '{game_id}' lancé pour {player_name} — bonne partie !", \
               {"marginTop": "12px", "fontSize": "12px", "color": "#00F5FF"}
    except Exception as e:
        return f"❌ Erreur : {e}", \
               {"marginTop": "12px", "fontSize": "12px", "color": "#FF4C6A"}

@app.callback(
    Output("live-inputs-table",   "children"),
    Output("live-joystick-graph", "figure"),
    Output("gauge-lt",            "children"),
    Output("gauge-rt",            "children"),
    Output("live-buttons-display","children"),
    Output("live-source-badge",   "children"),
    Input("refresh-interval", "n_intervals"),
    dash.dependencies.State("theme-store", "data"),
)
def update_live_inputs(n, theme):
    t    = THEMES[theme]
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
        "keyboard":   ("⌨️  Clavier", t["accent3"]),
    }
    label, color = badge_map.get(source, ("⏳ En attente…", t["subtext"]))
    badge = html.Div(label, style={"color": color, "fontSize": "10px", "letterSpacing": "1px",
                                    "border": f"1px solid {color}", "borderRadius": "4px",
                                    "padding": "2px 8px"})

    # ── Tableau (10 derniers) ──
    table = make_inputs_table(theme, rows[-10:] if rows else [])

    # ── Graphique joystick (60 derniers) ──
    if rows:
        lx_v = [r.get("lx", 0) for r in rows]
        ly_v = [r.get("ly", 0) for r in rows]
        ts   = list(range(len(rows)))
    else:
        ts   = list(range(60))
        lx_v = [np.sin(i * 0.2) * 0.5 for i in ts]
        ly_v = [np.cos(i * 0.15) * 0.4 for i in ts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ts, y=lx_v, name="LX",
                             line=dict(color=t["accent1"], width=2), mode="lines"))
    fig.add_trace(go.Scatter(x=ts, y=ly_v, name="LY",
                             line=dict(color=t["accent2"], width=2), mode="lines"))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=t["text"], family=t["font_body"]),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridcolor=t["border"], range=[-1.1, 1.1]),
        legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=1.1),
        margin=dict(l=20, r=20, t=10, b=10), height=180,
    )

    # ── Jauges ──
    last   = rows[-1] if rows else {}
    lt_val = float(last.get("lt", 0) or 0)
    rt_val = float(last.get("rt", 0) or 0)
    gauge_lt = html.Div(style={"width": f"{int(lt_val*100)}%", "height": "8px",
                                "background": t["accent2"], "borderRadius": "4px",
                                "transition": "width 0.3s ease", "minWidth": "4px"})
    gauge_rt = html.Div(style={"width": f"{int(rt_val*100)}%", "height": "8px",
                                "background": t["accent1"], "borderRadius": "4px",
                                "transition": "width 0.3s ease", "minWidth": "4px"})

    # ── Boutons ──
    btns       = {"A": last.get("btn_a", False), "B": last.get("btn_b", False),
                  "X": last.get("btn_x", False), "Y": last.get("btn_y", False)}
    btn_colors = {"A": "#69FF47", "B": "#FF4C6A", "X": "#00E5FF", "Y": "#FFB800"}
    buttons_display = html.Div([
        html.Div(btn, style={
            "width": "36px", "height": "36px", "borderRadius": "50%",
            "display": "flex", "alignItems": "center", "justifyContent": "center",
            "fontFamily": t["font"], "fontSize": "12px", "fontWeight": "700",
            "background": btn_colors[btn] if pressed else t["border"],
            "color": "#000" if pressed else t["subtext"],
            "transition": "all 0.1s ease",
            "boxShadow": f"0 0 10px {btn_colors[btn]}" if pressed else "none",
        })
        for btn, pressed in btns.items()
    ], style={"display": "flex", "gap": "8px"})

    return table, fig, gauge_lt, gauge_rt, buttons_display, badge   




# ─────────────────────────────────────────────────────────────────────────────
# PAGE RÉSUMÉS LLM
# ─────────────────────────────────────────────────────────────────────────────

def page_summary(theme):
    t = THEMES[theme]
    return html.Div([
        # Titre
        html.Div([
            html.Div([
                html.Div("📋 Résumés de sessions", style={
                    "color": t["accent1"], "fontSize": "22px",
                    "fontWeight": "700", "fontFamily": t["font"], "marginBottom": "4px"
                }),
                html.Div("Analyse IA personnalisée après chaque partie",
                         style={"color": t["subtext"], "fontSize": "13px"}),
            ]),
        ], style={"marginBottom": "24px"}),

        # Résumé le plus récent en avant
        make_card([
            html.Div("Dernière session analysée", style={
                "color": t["subtext"], "fontSize": "11px",
                "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "16px"
            }),
            html.Div(id="summary-latest"),
        ], theme, {"marginBottom": "16px"}),

        # Historique
        make_card([
            html.Div("Historique des résumés", style={
                "color": t["subtext"], "fontSize": "11px",
                "textTransform": "uppercase", "letterSpacing": "2px", "marginBottom": "16px"
            }),
            html.Div(id="summary-history"),
        ], theme),
    ])


def _render_summary_card(s: dict, t: dict, compact: bool = False) -> html.Div:
    """Construit le rendu HTML d'un résumé de session."""
    summary = s.get("summary_json", {}) or {}
    if not summary:
        return html.Div()

    emoji         = summary.get("emoji_humeur", "🎮")
    titre         = summary.get("titre", "Session")
    resume        = summary.get("resume", "")
    analyse_style = summary.get("analyse_style", "")
    profil_joueur = summary.get("profil_joueur", "")
    conseil       = summary.get("conseil", "")
    objectif      = summary.get("objectif", "")
    cl_glob       = summary.get("classement_global", "")
    cl_perso      = summary.get("classement_personnel", "")
    pf            = summary.get("points_forts", [])
    axes          = summary.get("axes_amelioration", [])
    is_mock       = summary.get("mock", False)

    game_colors = {
        "reflex": t["accent1"], "labyrinth": t["accent2"],
        "shooter": t["accent3"], "racing": "#FFB800",
    }
    game_col = game_colors.get(s.get("game_id", ""), t["subtext"])

    if compact:
        return html.Div([
            html.Div([
                html.Span(emoji, style={"fontSize": "18px", "marginRight": "8px"}),
                html.Span(titre, style={"color": t["text"], "fontWeight": "700",
                                        "fontFamily": t["font"], "fontSize": "13px"}),
                html.Span(f" · {s.get('player_name','')}",
                          style={"color": t["subtext"], "fontSize": "11px"}),
                html.Span(f" · {s.get('game_id','').upper()}",
                          style={"color": game_col, "fontSize": "11px", "marginLeft": "4px"}),
                html.Span(f" · {s.get('score', 0)} pts",
                          style={"color": t["accent2"], "fontSize": "11px", "marginLeft": "4px"}),
                *([html.Span(f" · {profil_joueur}",
                             style={"color": t["accent1"], "fontSize": "10px",
                                    "marginLeft": "4px", "fontStyle": "italic"})] if profil_joueur else []),
            ], style={"marginBottom": "4px"}),
            html.Div(resume, style={"color": t["subtext"], "fontSize": "12px",
                                     "borderLeft": f"2px solid {t['border']}",
                                     "paddingLeft": "8px", "marginBottom": "8px",
                                     "lineHeight": "1.5"}),
        ], style={"marginBottom": "12px", "paddingBottom": "12px",
                  "borderBottom": f"1px solid {t['border']}"})

    # ── Version complète ────────────────────────────────────────────────────
    return html.Div([

        # Header : emoji + titre + meta
        html.Div([
            html.Span(emoji, style={"fontSize": "40px", "marginRight": "14px", "lineHeight": "1"}),
            html.Div([
                html.Div(titre, style={"color": t["text"], "fontWeight": "700",
                                       "fontFamily": t["font"], "fontSize": "18px",
                                       "lineHeight": "1.2", "marginBottom": "4px"}),
                html.Div([
                    html.Span(s.get("player_name", ""), style={"color": t["accent1"], "fontWeight": "600"}),
                    html.Span(" · ", style={"color": t["subtext"]}),
                    html.Span(s.get("game_id", "").upper(), style={"color": game_col}),
                    html.Span(" · ", style={"color": t["subtext"]}),
                    html.Span(f"{s.get('score', 0)} pts", style={"color": t["accent2"], "fontWeight": "600"}),
                    html.Span(" · ", style={"color": t["subtext"]}),
                    html.Span(f"{s.get('duration_sec', 0):.0f}s", style={"color": t["subtext"]}),
                ], style={"fontSize": "12px"}),
            ]),
            # Badge profil
            *([html.Div(profil_joueur, style={
                "marginLeft": "auto", "background": t["accent1"] + "22",
                "border": f"1px solid {t['accent1']}", "color": t["accent1"],
                "borderRadius": "20px", "padding": "4px 12px",
                "fontSize": "11px", "fontWeight": "700", "whiteSpace": "nowrap",
                "fontFamily": t["font"],
            })] if profil_joueur else []),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "16px",
                  "flexWrap": "wrap", "gap": "8px"}),

        # Mock warning
        *([html.Div("⚠ Résumé généré localement (API Mistral indisponible)", style={
            "background": "#FFB80022", "border": "1px solid #FFB800", "color": "#FFB800",
            "borderRadius": "6px", "padding": "8px 12px", "fontSize": "11px",
            "marginBottom": "12px",
        })] if is_mock else []),

        # Résumé principal
        html.Div([
            html.Div("Résumé", style={"color": t["subtext"], "fontSize": "10px",
                                       "textTransform": "uppercase", "letterSpacing": "2px",
                                       "marginBottom": "6px"}),
            html.Div(resume, style={"color": t["text"], "fontSize": "13px",
                                     "lineHeight": "1.7"}),
        ], style={"background": t["bg"], "borderRadius": "6px", "padding": "12px 16px",
                  "marginBottom": "12px", "borderLeft": f"3px solid {t['accent2']}"}),

        # Analyse du style de jeu
        *([html.Div([
            html.Div("Analyse du style de jeu", style={"color": t["subtext"], "fontSize": "10px",
                                                         "textTransform": "uppercase",
                                                         "letterSpacing": "2px", "marginBottom": "6px"}),
            html.Div(analyse_style, style={"color": t["text"], "fontSize": "13px",
                                            "lineHeight": "1.7"}),
        ], style={"background": t["bg"], "borderRadius": "6px", "padding": "12px 16px",
                  "marginBottom": "12px", "borderLeft": f"3px solid {t['accent1']}"})] if analyse_style else []),

        # Classements
        html.Div([
            html.Div([
                html.Span("🏆 ", style={"fontSize": "14px"}),
                html.Span(cl_glob, style={"color": t["accent2"], "fontSize": "12px",
                                           "lineHeight": "1.5"}),
            ], style={"marginBottom": "6px"}),
            html.Div([
                html.Span("📊 ", style={"fontSize": "14px"}),
                html.Span(cl_perso, style={"color": t["accent3"], "fontSize": "12px",
                                            "lineHeight": "1.5"}),
            ]),
        ], style={"background": t["bg"], "borderRadius": "6px", "padding": "12px 16px",
                  "marginBottom": "12px"}),

        # Points forts + axes (côte à côte)
        html.Div([
            html.Div([
                html.Div("💪 Points forts", style={"color": t["accent2"], "fontSize": "11px",
                                                    "fontWeight": "700", "letterSpacing": "1px",
                                                    "marginBottom": "8px"}),
                *[html.Div([
                    html.Span("✓ ", style={"color": t["accent2"], "fontWeight": "700"}),
                    html.Span(p, style={"color": t["text"], "fontSize": "12px"}),
                ], style={"marginBottom": "6px", "lineHeight": "1.4"}) for p in pf],
            ], style={"flex": "1", "background": t["bg"], "borderRadius": "6px",
                      "padding": "12px", "borderTop": f"2px solid {t['accent2']}"}),
            html.Div([
                html.Div("📈 Axes d'amélioration", style={"color": t["accent3"], "fontSize": "11px",
                                                           "fontWeight": "700", "letterSpacing": "1px",
                                                           "marginBottom": "8px"}),
                *[html.Div([
                    html.Span("→ ", style={"color": t["accent3"], "fontWeight": "700"}),
                    html.Span(a, style={"color": t["text"], "fontSize": "12px"}),
                ], style={"marginBottom": "6px", "lineHeight": "1.4"}) for a in axes],
            ], style={"flex": "1", "background": t["bg"], "borderRadius": "6px",
                      "padding": "12px", "borderTop": f"2px solid {t['accent3']}"}),
        ], style={"display": "flex", "gap": "12px", "marginBottom": "12px"}),

        # Conseil
        html.Div([
            html.Div([
                html.Span("💡 ", style={"fontSize": "16px"}),
                html.Span("Conseil coach : ", style={"color": t["accent1"], "fontWeight": "700",
                                                      "fontSize": "12px"}),
                html.Span(conseil, style={"color": t["text"], "fontSize": "12px",
                                           "lineHeight": "1.5"}),
            ], style={"marginBottom": "8px" if objectif else "0"}),
            *([html.Div([
                html.Span("🎯 ", style={"fontSize": "16px"}),
                html.Span("Objectif : ", style={"color": "#FFB800", "fontWeight": "700",
                                                 "fontSize": "12px"}),
                html.Span(objectif, style={"color": t["text"], "fontSize": "12px",
                                            "lineHeight": "1.5"}),
            ])] if objectif else []),
        ], style={"background": t["bg"], "borderRadius": "6px", "padding": "12px 16px",
                  "borderLeft": f"3px solid {t['accent1']}"}),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS RÉSUMÉS
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("summary-store", "data"),
    Input("summary-refresh-interval", "n_intervals"),
)
def refresh_summaries(n):
    """Rafraîchit les résumés depuis Supabase toutes les 8s."""
    try:
        from core.llm_summary import fetch_latest_summaries
        return fetch_latest_summaries(limit=20)
    except Exception:
        return []


@app.callback(
    Output("summary-latest",  "children"),
    Output("summary-history", "children"),
    Input("summary-store", "data"),
    dash.dependencies.State("theme-store", "data"),
)
def render_summaries(summaries, theme):
    t = THEMES[theme]

    if not summaries:
        empty = html.Div([
            html.Div("Aucun résumé disponible", style={
                "color": t["subtext"], "fontSize": "13px", "textAlign": "center",
                "padding": "32px",
            }),
            html.Div("Lance une partie pour générer ton premier résumé IA !",
                     style={"color": t["subtext"], "fontSize": "11px",
                            "textAlign": "center", "fontStyle": "italic"}),
        ])
        return empty, empty

    # Dernière session
    latest  = _render_summary_card(summaries[0], t, compact=False)

    # Historique (sans la première)
    if len(summaries) > 1:
        history = html.Div([
            _render_summary_card(s, t, compact=True) for s in summaries[1:]
        ])
    else:
        history = html.Div("Lance d'autres parties pour voir l'historique.",
                           style={"color": t["subtext"], "fontSize": "12px",
                                  "fontStyle": "italic"})

    return latest, history


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS AGENT IA
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("dropdown-agent-player", "options"),
    Input("dropdown-agent-game", "value"),
)
def update_agent_players(game_id):
    """Peuple la liste des joueurs disponibles pour le jeu sélectionné."""
    if not game_id:
        return [{"label": p, "value": p} for p in mock_players]
    try:
        from core.supabase_client import fetch_all_sessions
        sessions = fetch_all_sessions()
        players = sorted({
            s["player_name"] for s in sessions
            if s.get("game_id") == game_id
            and s.get("player_name")
            and not s["player_name"].startswith("Agent_")
        })
        if players:
            return [{"label": p, "value": p} for p in players]
    except Exception:
        pass
    return [{"label": p, "value": p} for p in mock_players]


@app.callback(
    Output("agent-feedback", "children"),
    Output("agent-feedback", "style"),
    Output("agent-pid-store", "data"),
    Input("btn-launch-agent", "n_clicks"),
    dash.dependencies.State("dropdown-agent-game",   "value"),
    dash.dependencies.State("dropdown-agent-player", "value"),
    dash.dependencies.State("slider-agent-noise",    "value"),
    prevent_initial_call=True,
)
def launch_agent(n_clicks, game_id, player_name, fidelity):
    """Lance l'agent IA en subprocess qui rejoue les inputs du joueur sélectionné."""
    if not game_id:
        return "⚠️ Sélectionne un jeu.", {"marginTop": "12px", "fontSize": "12px", "color": "#FFB800"}, None
    if not player_name:
        return "⚠️ Sélectionne un joueur à imiter.", {"marginTop": "12px", "fontSize": "12px", "color": "#FFB800"}, None

    # Fidélité 0–100 → noise_level 2.0–0.0
    noise = round((100 - (fidelity or 80)) / 50, 2)
    agent_session_name = f"Agent_{player_name}"

    try:
        proc = subprocess.Popen(
            [
                sys.executable, "main.py",
                game_id, agent_session_name,
                "--agent", player_name,
                "--mode", "player",
                "--noise", str(noise),
            ],
            cwd=ROOT_DIR,
        )
        msg   = f"🤖 Agent '{player_name}' lancé sur {game_id} (fidélité {fidelity}%)"
        style = {"marginTop": "12px", "fontSize": "12px", "color": "#00F5FF"}
        return msg, style, proc.pid
    except Exception as e:
        return f"❌ Erreur : {e}", {"marginTop": "12px", "fontSize": "12px", "color": "#FF4C6A"}, None


@app.callback(
    Output("agent-feedback", "children", allow_duplicate=True),
    Output("agent-feedback", "style",    allow_duplicate=True),
    Output("agent-pid-store", "data",    allow_duplicate=True),
    Input("btn-stop-agent", "n_clicks"),
    dash.dependencies.State("agent-pid-store", "data"),
    prevent_initial_call=True,
)
def stop_agent(n_clicks, pid):
    """Arrête le subprocess de l'agent en cours."""
    if not pid:
        return "⚠️ Aucun agent en cours.", {"marginTop": "12px", "fontSize": "12px", "color": "#FFB800"}, None
    try:
        subprocess.call(["taskkill", "/F", "/T", "/PID", str(pid)])
        return "⏹ Agent arrêté.", {"marginTop": "12px", "fontSize": "12px", "color": "#FF4C6A"}, None
    except Exception as e:
        return f"❌ Erreur arrêt : {e}", {"marginTop": "12px", "fontSize": "12px", "color": "#FF4C6A"}, None


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS POST-SESSION & CHAT
# ─────────────────────────────────────────────────────────────────────────────

@app.callback(
    Output("url-params-store",      "data"),
    Output("page-store",            "data",     allow_duplicate=True),
    Output("postsession-interval",  "disabled"),
    Input("url", "search"),
    prevent_initial_call=True,
)
def parse_url(search):
    """Lit les query params de l'URL (?player=X&game=Y&ts=T) et navigue vers post-session."""
    if not search:
        return {}, dash.no_update, True
    try:
        from urllib.parse import parse_qs
        params = parse_qs(search.lstrip("?"))
        player = params.get("player", [None])[0]
        game   = params.get("game",   [None])[0]
        ts     = params.get("ts",     [None])[0]
        if player and game:
            store = {"player": player, "game": game}
            if ts:
                store["ts"] = int(ts)
            return store, "postsession", False
    except Exception:
        pass
    return {}, dash.no_update, True


@app.callback(
    Output("postsession-summary-data",  "data"),
    Output("postsession-interval", "disabled", allow_duplicate=True),
    Input("postsession-interval", "n_intervals"),
    dash.dependencies.State("url-params-store", "data"),
    prevent_initial_call=True,
)
def load_postsession_summary(n, params):
    """Polling toutes les 2s jusqu'à ce que le résumé LLM soit disponible."""
    player = (params or {}).get("player", "")
    game   = (params or {}).get("game",   "")
    ts_min = (params or {}).get("ts", 0)   # timestamp minimum accepté
    if not player or not game:
        return dash.no_update, True

    try:
        from core.llm_summary import fetch_latest_summaries
        import datetime, time as _t
        summaries = fetch_latest_summaries(limit=30)
        for s in summaries:
            if s.get("player_name") != player or s.get("game_id") != game:
                continue
            # Si on a un ts minimum, vérifier que le résumé est récent
            if ts_min:
                created = s.get("created_at", "")
                try:
                    # Supabase renvoie ISO 8601 : "2025-03-05T12:34:56.123+00:00"
                    from datetime import timezone
                    dt = datetime.datetime.fromisoformat(created.replace("Z", "+00:00"))
                    if dt.timestamp() < ts_min - 5:   # tolérance 5s
                        continue
                except Exception:
                    pass   # Si on ne peut pas parser, on accepte quand même
            return s, True   # Résumé trouvé → stop polling
    except Exception:
        pass

    return dash.no_update, False


@app.callback(
    Output("chat-messages-container", "children"),
    Input("chat-store", "data"),
    dash.dependencies.State("theme-store", "data"),
)
def render_chat_messages(history, theme):
    """Affiche les messages du chatbot."""
    t = THEMES[theme]
    if not history:
        return [html.Div([
            html.Span("🤖", style={"fontSize": "18px", "marginRight": "10px"}),
            html.Span("Bonjour ! Je suis ton coach IA. Pose-moi une question sur tes performances.",
                      style={"color": t["text"], "fontSize": "13px"}),
        ], style={"background": t["bg"], "borderRadius": "8px", "padding": "12px 16px",
                  "marginBottom": "8px", "borderLeft": f"3px solid {t['accent1']}"})]

    messages = []
    for msg in history:
        is_user = msg["role"] == "user"
        messages.append(html.Div([
            html.Span("👤" if is_user else "🤖",
                      style={"fontSize": "16px", "marginRight": "8px"}),
            html.Span(msg["content"],
                      style={"color": t["text"], "fontSize": "13px", "lineHeight": "1.5"}),
        ], style={
            "background":   t["bg"] if not is_user else t["card"],
            "borderRadius": "8px",
            "padding":      "10px 14px",
            "marginBottom": "8px",
            "borderLeft":   f"3px solid {t['accent2'] if is_user else t['accent1']}",
            "marginLeft":   "20px" if is_user else "0",
        }))
    return messages


@app.callback(
    Output("chat-store",   "data"),
    Output("chat-input",   "value"),
    Output("chat-loading", "children"),
    Input("btn-chat-send", "n_clicks"),
    Input("chat-input",    "n_submit"),
    dash.dependencies.State("chat-input",      "value"),
    dash.dependencies.State("chat-store",      "data"),
    dash.dependencies.State("sessions-store",  "data"),
    prevent_initial_call=True,
)
def send_chat_message(n_clicks, n_submit, message, history, sessions_data):
    """Envoie un message au chatbot Mistral avec contexte sessions."""
    if not message or not message.strip():
        return dash.no_update, dash.no_update, ""

    history = history or []

    # ── Construction du contexte structuré ───────────────────────────────────
    context_parts = []
    if sessions_data:
        df_ctx = pd.DataFrame(sessions_data)

        # 1) Classement par joueur (score max + score moyen) par jeu
        if "game_id" in df_ctx.columns and "score" in df_ctx.columns:
            for game in sorted(df_ctx["game_id"].dropna().unique()):
                gdf = df_ctx[df_ctx["game_id"] == game].copy()
                player_rank = (
                    gdf.groupby("player_name")["score"]
                    .agg(best="max", mean="mean", sessions="count")
                    .sort_values("best", ascending=False)
                    .reset_index()
                )
                context_parts.append(
                    f"\n=== CLASSEMENT {game.upper()} ({len(gdf)} sessions, "
                    f"{len(player_rank)} joueurs) ==="
                )
                for i, row in player_rank.iterrows():
                    context_parts.append(
                        f"  #{i+1} {row['player_name']}"
                        f"  meilleur={int(row['best'])}"
                        f"  moy={int(row['mean'])}"
                        f"  sessions={int(row['sessions'])}"
                    )

            # 2) Top 5 sessions par jeu (triées par score desc)
            for game in sorted(df_ctx["game_id"].dropna().unique()):
                gdf = df_ctx[df_ctx["game_id"] == game].nlargest(5, "score")
                context_parts.append(f"\n--- TOP 5 SESSIONS {game.upper()} ---")
                for _, row in gdf.iterrows():
                    reg = f"  reg={row.get('input_regularity',0):.2f}" if "input_regularity" in row else ""
                    context_parts.append(
                        f"  {row.get('player_name','?')}"
                        f"  score={int(row.get('score',0))}"
                        f"  durée={int(row.get('duration_sec',0))}s"
                        f"  btn_rate={row.get('btn_press_rate',0):.3f}"
                        f"{reg}"
                    )

            # 3) Stats globales
            total = len(df_ctx)
            best_row = df_ctx.loc[df_ctx["score"].idxmax()]
            context_parts.append(
                f"\n=== STATS GLOBALES ==="
                f"\n  Total sessions : {total}"
                f"\n  Joueurs uniques : {df_ctx['player_name'].nunique()}"
                f"\n  Meilleur score absolu : {int(best_row.get('score',0))}"
                f" par {best_row.get('player_name','?')} sur {best_row.get('game_id','?')}"
            )

    context = "\n".join(context_parts)

    # ── Appel LLM ─────────────────────────────────────────────────────────────
    try:
        from core.llm_summary import chat_with_llm
        response = chat_with_llm(message.strip(), history, context)
    except Exception as e:
        response = f"Erreur : {e}"

    new_history = history + [
        {"role": "user",      "content": message.strip()},
        {"role": "assistant", "content": response},
    ]
    return new_history, "", ""


if __name__ == "__main__":
    app.run(debug=True, port=8050)


