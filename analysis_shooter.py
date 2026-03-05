#!/usr/bin/env python3
"""
analysis_shooter.py — Analyse comportementale des sessions Shooter
====================================================================
Utilisation standalone : python analysis_shooter.py
Importé par app_new.py  : from analysis_shooter import compute_shooter_analysis
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── UMAP optionnel ────────────────────────────────────────────────────────────
try:
    import umap as umap_lib
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("⚠️  umap-learn non installé — PCA utilisée à la place")

# ── Features comportementales (on exclut score/durée/meta) ───────────────────
FEATURE_COLS = [
    "btn_press_rate", "btn_variety", "btn_hold_avg_ms",
    "lx_mean", "ly_mean", "lx_std", "ly_std", "lx_direction_changes",
    "rx_std", "ry_std", "rt_mean", "lt_mean", "input_regularity",
]

FEATURE_LABELS = {
    "btn_press_rate":       "Pression boutons",
    "btn_variety":          "Variété boutons",
    "btn_hold_avg_ms":      "Durée appui (ms)",
    "lx_mean":              "Joystick G. (X moyen)",
    "ly_mean":              "Joystick G. (Y moyen)",
    "lx_std":               "Agitation joystick G.",
    "ly_std":               "Agitation joystick G. (Y)",
    "lx_direction_changes": "Changements direction",
    "rx_std":               "Agitation visée (X)",
    "ry_std":               "Agitation visée (Y)",
    "rt_mean":              "Gâchette tir",
    "lt_mean":              "Gâchette gauche",
    "input_regularity":     "Régularité",
}

# ── Règles de nommage automatique des clusters (sur centroïdes normalisés) ───
_NAME_RULES = [
    ("rt_mean",         ">",  0.45, "🔥 Tireur Agressif"),
    ("input_regularity","<", -0.35, "🎲 Style Chaotique"),
    ("lx_std",          ">",  0.45, "🏃 Mobile Dynamique"),
    ("input_regularity",">",  0.45, "⚡ Précis Régulier"),
    ("lx_std",          "<", -0.40, "🛡️ Défensif Statique"),
    ("btn_press_rate",  ">",  0.50, "💥 Boutonneur Actif"),
]

FALLBACK_NAMES = ["🔴 Cluster Alpha", "🔵 Cluster Bêta", "🟢 Cluster Gamma", "🟡 Cluster Delta"]


def _auto_name_cluster(centroid: pd.Series, used: set) -> str:
    for feat, op, thr, name in _NAME_RULES:
        if feat not in centroid or name in used:
            continue
        val = centroid[feat]
        if (op == ">" and val > thr) or (op == "<" and val < thr):
            return name
    for name in FALLBACK_NAMES:
        if name not in used:
            return name
    return "🔵 Cluster"


# ══════════════════════════════════════════════════════════════════════════════
# FONCTION CENTRALE : compute_shooter_analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_shooter_analysis(df_raw: pd.DataFrame) -> dict | None:
    """
    Calcule les 3 analyses comportementales sur les sessions shooter.

    Paramètre
    ---------
    df_raw : DataFrame avec colonnes SESSION_COLUMNS + eventuellement 'game_id'

    Retourne
    --------
    dict avec clés "clustering", "progression", "correlation", "df", "features"
    ou None si données insuffisantes.
    """
    # Filtre shooter
    if "game_id" in df_raw.columns:
        df = df_raw[df_raw["game_id"] == "shooter"].copy()
    else:
        df = df_raw.copy()

    if df.empty or len(df) < 5:
        return None

    avail = [f for f in FEATURE_COLS if f in df.columns]
    if len(avail) < 3:
        return None

    X_raw = df[avail].fillna(0)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    # ─── Analyse 1 : Clustering ───────────────────────────────────────────────
    sil_scores = {}
    for k in [2, 3, 4]:
        try:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            lbl = km.fit_predict(X)
            if len(set(lbl)) > 1:
                sil_scores[k] = round(silhouette_score(X, lbl), 4)
        except Exception:
            pass

    best_k = max(sil_scores, key=sil_scores.get) if sil_scores else 3

    km3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_labels = km3.fit_predict(X)

    # PCA 2D
    pca = PCA(n_components=min(2, X.shape[1]))
    pca_coords = pca.fit_transform(X)
    pca_var = pca.explained_variance_ratio_

    # UMAP 2D
    umap_coords = None
    if UMAP_AVAILABLE and len(df) >= 10:
        try:
            reducer = umap_lib.UMAP(n_neighbors=min(10, len(df) - 1),
                                    min_dist=0.3, random_state=42)
            umap_coords = reducer.fit_transform(X)
        except Exception as e:
            print(f"⚠️  UMAP échoué : {e}")

    # Centroïdes normalisés → nommage auto
    centroids_norm = pd.DataFrame(km3.cluster_centers_, columns=avail)
    cluster_names = {}
    used = set()
    for c in range(3):
        name = _auto_name_cluster(centroids_norm.iloc[c], used)
        cluster_names[c] = name
        used.add(name)

    # Moyennes réelles par cluster
    df_cl = df.copy()
    df_cl["cluster"] = cluster_labels
    display_feats = [f for f in ["rt_mean", "lx_std", "btn_press_rate",
                                  "input_regularity", "rx_std", "lt_mean",
                                  "lx_direction_changes"] if f in df.columns]
    centroids_real = df_cl.groupby("cluster")[display_feats + (["score"] if "score" in df.columns else [])].mean().round(3)
    centroids_real.insert(0, "Profil", [cluster_names.get(i, str(i)) for i in centroids_real.index])
    centroids_real["Sessions"] = df_cl.groupby("cluster").size().values

    clustering = {
        "labels":           cluster_labels,
        "player_names":     df["player_name"].values if "player_name" in df.columns else np.full(len(df), "?"),
        "pca_xy":           pca_coords,
        "umap_xy":          umap_coords,
        "cluster_names":    cluster_names,
        "centroids_norm":   centroids_norm,
        "centroids_real":   centroids_real,
        "silhouette_scores":sil_scores,
        "best_k":           best_k,
        "pca_var":          pca_var,
        "n":                len(df),
    }

    # ─── Analyse 2 : Progression intra-joueur ────────────────────────────────
    players, slopes, r2_list, pvalues, statuses = [], [], [], [], []
    sessions_data, trend_lines = {}, {}

    if "player_name" in df.columns and "score" in df.columns:
        df_s = df.sort_values("created_at").reset_index(drop=True) if "created_at" in df.columns else df.reset_index(drop=True)
        for player, grp in df_s.groupby("player_name"):
            sc = grp["score"].values
            if len(sc) < 2:
                continue
            x = np.arange(len(sc))
            res = stats.linregress(x, sc)
            slope, intercept, r_val, p_val, _ = res
            if p_val < 0.1:
                status = "📈 En progression" if slope > 0 else "📉 En régression"
            else:
                status = "➡️ Stable"
            players.append(player)
            slopes.append(round(slope, 3))
            r2_list.append(round(r_val ** 2, 4))
            pvalues.append(round(p_val, 4))
            statuses.append(status)
            sessions_data[player] = list(zip(x, sc))
            trend_lines[player]   = (x, intercept + slope * x)

    # Tri par pente décroissante
    if players:
        order = np.argsort(slopes)[::-1]
        players  = [players[i]  for i in order]
        slopes   = [slopes[i]   for i in order]
        r2_list  = [r2_list[i]  for i in order]
        pvalues  = [pvalues[i]  for i in order]
        statuses = [statuses[i] for i in order]

    progression = {
        "players":       players,
        "slopes":        np.array(slopes),
        "r2":            np.array(r2_list),
        "pvalues":       np.array(pvalues),
        "statuses":      statuses,
        "sessions_data": sessions_data,
        "trend_lines":   trend_lines,
    }

    # ─── Analyse 3 : Corrélation features → score ────────────────────────────
    corr_features, spearman_r, corr_pvals = [], [], []

    if "score" in df.columns:
        y = df["score"].fillna(0).values
        for feat in avail:
            x = df[feat].fillna(0).values
            r, p = stats.spearmanr(x, y)
            corr_features.append(feat)
            spearman_r.append(round(r, 4))
            corr_pvals.append(round(p, 4))
        order = np.argsort(np.abs(spearman_r))[::-1]
        corr_features = [corr_features[i] for i in order]
        spearman_r    = [spearman_r[i]    for i in order]
        corr_pvals    = [corr_pvals[i]    for i in order]

    top3 = corr_features[:3] if corr_features else []

    correlation = {
        "features":   corr_features,
        "spearman_r": np.array(spearman_r),
        "pvalues":    np.array(corr_pvals),
        "top3":       top3,
        "df":         df,
    }

    return {
        "clustering":  clustering,
        "progression": progression,
        "correlation": correlation,
        "df":          df,
        "features":    avail,
    }


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATIONS MATPLOTLIB (standalone)
# ══════════════════════════════════════════════════════════════════════════════

def _setup_dark():
    import matplotlib.pyplot as plt
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor":  "#0A0A0F",
        "axes.facecolor":    "#12121F",
        "axes.edgecolor":    "#7B2FBE",
        "axes.labelcolor":   "#E8E8FF",
        "text.color":        "#E8E8FF",
        "xtick.color":       "#8888AA",
        "ytick.color":       "#8888AA",
        "grid.color":        "#2A2A3F",
        "grid.alpha":        0.5,
        "axes.titlesize":    12,
        "axes.labelsize":    10,
        "font.family":       "monospace",
    })
    return plt


CLUSTER_COLORS = ["#C724B1", "#00F5FF", "#69FF47", "#FFB800"]
STATUS_COLORS  = {"📈 En progression": "#69FF47", "➡️ Stable": "#00F5FF", "📉 En régression": "#FF4C6A"}


def plot_clustering(result: dict, save_path: str = None):
    plt = _setup_dark()
    import matplotlib.pyplot as mpl_plt
    cl  = result["clustering"]
    df  = result["df"]

    fig, axes = mpl_plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Analyse 1 — Clustering Comportemental Shooter", fontsize=14,
                 color="#C724B1", fontweight="bold")

    coords = cl["umap_xy"] if cl["umap_xy"] is not None else cl["pca_xy"]
    proj_label = "UMAP" if cl["umap_xy"] is not None else "PCA"

    # ── Scatter par cluster ──
    ax = axes[0, 0]
    for c, name in cl["cluster_names"].items():
        mask = cl["labels"] == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=CLUSTER_COLORS[c % 4], label=name, s=80, alpha=0.85,
                   edgecolors="#0A0A0F", linewidths=0.5)
    ax.set_title(f"{proj_label} 2D · coloré par cluster")
    ax.legend(fontsize=8, framealpha=0.3)
    ax.grid(True)

    # ── Scatter par joueur ──
    ax = axes[0, 1]
    players_u = list(set(cl["player_names"]))
    cmap = mpl_plt.get_cmap("tab20", len(players_u))
    for i, player in enumerate(players_u):
        mask = cl["player_names"] == player
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[cmap(i)], label=player, s=70, alpha=0.85,
                   edgecolors="#0A0A0F", linewidths=0.5)
    ax.set_title(f"{proj_label} 2D · coloré par joueur")
    ax.legend(fontsize=7, framealpha=0.3, ncol=2)
    ax.grid(True)

    # ── Silhouette scores ──
    ax = axes[1, 0]
    ks = list(cl["silhouette_scores"].keys())
    sils = list(cl["silhouette_scores"].values())
    bars = ax.bar(ks, sils, color=CLUSTER_COLORS[:len(ks)], alpha=0.85, edgecolor="#0A0A0F")
    ax.bar_label(bars, fmt="%.3f", color="#E8E8FF", fontsize=9)
    ax.axvline(x=3, color="#FFB800", linestyle="--", alpha=0.7, label="k=3 (choisi)")
    ax.set_title("Silhouette Score par k")
    ax.set_xlabel("Nombre de clusters k")
    ax.set_ylabel("Silhouette Score")
    ax.legend(fontsize=8)
    ax.set_xticks(ks)
    ax.grid(True, axis="y")

    # ── Heatmap centroïdes normalisés ──
    ax = axes[1, 1]
    import matplotlib.cm as cm
    c_data = cl["centroids_norm"].values
    feat_labs = [FEATURE_LABELS.get(f, f)[:14] for f in cl["centroids_norm"].columns]
    im = ax.imshow(c_data, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(len(feat_labs)))
    ax.set_xticklabels(feat_labs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(3))
    ax.set_yticklabels([cl["cluster_names"].get(i, str(i)) for i in range(3)], fontsize=9)
    ax.set_title("Centroïdes normalisés (rouge=haut, bleu=bas)")
    for r in range(3):
        for cc in range(len(feat_labs)):
            ax.text(cc, r, f"{c_data[r, cc]:.2f}", ha="center", va="center",
                    fontsize=6, color="white")
    mpl_plt.colorbar(im, ax=ax, shrink=0.8)

    mpl_plt.tight_layout()
    if save_path:
        mpl_plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Clustering sauvegardé : {save_path}")
    else:
        mpl_plt.show()


def plot_progression(result: dict, save_path: str = None):
    plt = _setup_dark()
    import matplotlib.pyplot as mpl_plt
    pr = result["progression"]

    if not pr["players"]:
        print("⚠️  Pas assez de sessions pour la progression.")
        return

    n_players = len(pr["players"])
    ncols = min(4, n_players)
    nrows_grid = (n_players + ncols - 1) // ncols

    fig = mpl_plt.figure(figsize=(14, 4 * nrows_grid + 4))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(nrows_grid + 1, ncols, figure=fig, hspace=0.5, wspace=0.35)
    fig.suptitle("Analyse 2 — Progression Intra-Joueur", fontsize=14,
                 color="#00F5FF", fontweight="bold")

    for i, player in enumerate(pr["players"]):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])
        data = pr["sessions_data"].get(player, [])
        if not data:
            continue
        xs, ys = zip(*data)
        status = pr["statuses"][i]
        col_line = STATUS_COLORS.get(status, "#E8E8FF")
        ax.plot(xs, ys, "o-", color=col_line, markersize=5, linewidth=1.8, alpha=0.9)
        tx, ty = pr["trend_lines"].get(player, (xs, ys))
        ax.plot(tx, ty, "--", color="#FFB800", linewidth=1.2, alpha=0.7)
        ax.set_title(f"{player}\n{status}", fontsize=8, color=col_line)
        ax.set_xlabel("Session #", fontsize=7)
        ax.set_ylabel("Score", fontsize=7)
        ax.grid(True, alpha=0.4)
        pv = pr["pvalues"][i]
        sl = pr["slopes"][i]
        ax.text(0.05, 0.92, f"p={pv:.2f}  β={sl:+.1f}", transform=ax.transAxes,
                fontsize=6.5, color="#8888AA")

    # Barplot des pentes
    ax_bar = fig.add_subplot(gs[nrows_grid, :])
    colors_bar = [STATUS_COLORS.get(s, "#E8E8FF") for s in pr["statuses"]]
    bars = ax_bar.barh(pr["players"], pr["slopes"], color=colors_bar, alpha=0.85,
                       edgecolor="#0A0A0F", height=0.6)
    ax_bar.axvline(0, color="#8888AA", linewidth=1)
    ax_bar.set_xlabel("Pente de progression (pts/session)", fontsize=9)
    ax_bar.set_title("Synthèse — Pente de progression par joueur", fontsize=10)
    ax_bar.bar_label(bars, fmt="%.2f", color="#E8E8FF", fontsize=8, padding=3)
    ax_bar.grid(True, axis="x", alpha=0.4)

    if save_path:
        mpl_plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Progression sauvegardée : {save_path}")
    else:
        mpl_plt.show()


def plot_correlation(result: dict, save_path: str = None):
    plt = _setup_dark()
    import matplotlib.pyplot as mpl_plt
    co  = result["correlation"]
    df  = co["df"]

    if not co["features"] or "score" not in df.columns:
        print("⚠️  Données insuffisantes pour la corrélation.")
        return

    fig, axes = mpl_plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Analyse 3 — Corrélation Features → Score", fontsize=14,
                 color="#69FF47", fontweight="bold")

    # ── Barplot Spearman ──
    ax = axes[0]
    feats_lab = [FEATURE_LABELS.get(f, f) for f in co["features"]]
    colors_r = ["#69FF47" if r >= 0 else "#FF4C6A" for r in co["spearman_r"]]
    bars = ax.barh(feats_lab[::-1], co["spearman_r"][::-1],
                   color=colors_r[::-1], alpha=0.85, edgecolor="#0A0A0F")
    ax.axvline(0, color="#8888AA", linewidth=1)
    for bar, p in zip(bars, co["pvalues"][::-1]):
        star = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if star:
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    star, va="center", fontsize=8, color="#FFB800")
    ax.set_title("Corrélation Spearman avec le score\n(* p<.05  ** p<.01  *** p<.001)", fontsize=9)
    ax.set_xlabel("Spearman r")
    ax.grid(True, axis="x", alpha=0.4)

    # ── Scatter top feature ──
    for ax_i, feat_i in enumerate([0, 1]):
        ax = axes[feat_i + 1]
        if feat_i >= len(co["features"]):
            break
        feat = co["features"][feat_i]
        x = df[feat].fillna(0).values
        y = df["score"].fillna(0).values
        ax.scatter(x, y, c="#C724B1", alpha=0.75, s=60, edgecolors="#0A0A0F")
        # Droite de régression
        m, b, *_ = stats.linregress(x, y)
        xline = np.linspace(x.min(), x.max(), 100)
        ax.plot(xline, m * xline + b, color="#FFB800", linewidth=2, linestyle="--")
        r = co["spearman_r"][feat_i]
        p = co["pvalues"][feat_i]
        label = FEATURE_LABELS.get(feat, feat)
        ax.set_title(f"{label}\nvs Score  (r={r:+.3f}, p={p:.3f})", fontsize=9)
        ax.set_xlabel(label, fontsize=8)
        ax.set_ylabel("Score", fontsize=8)
        ax.grid(True, alpha=0.4)

    mpl_plt.tight_layout()
    if save_path:
        mpl_plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"💾 Corrélation sauvegardée : {save_path}")
    else:
        mpl_plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — chargement données + exécution standalone
# ══════════════════════════════════════════════════════════════════════════════

def _load_data() -> pd.DataFrame:
    """Charge les données depuis Supabase ou fichier CSV de fallback."""
    # Tentative Supabase
    try:
        from core.supabase_client import fetch_all_sessions
        data = fetch_all_sessions()
        if data and len(data) >= 5:
            df = pd.DataFrame(data)
            print(f"✅ Supabase : {len(df)} sessions chargées ({df['game_id'].value_counts().to_dict()})")
            return df
    except Exception as e:
        print(f"⚠️  Supabase indisponible : {e}")

    # Fallback CSV
    csv_paths = [
        os.path.join(os.path.dirname(__file__), "data", "sessions.csv"),
        os.path.join(os.path.dirname(__file__), "data", "synthetic_sessions_500.csv"),
    ]
    for path in csv_paths:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"📂 Fallback CSV : {path} ({len(df)} lignes)")
            return df

    # Mock minimaliste si rien n'est disponible
    print("⚠️  Aucune source de données — génération mock")
    rng = np.random.default_rng(42)
    n = 60
    players = [f"Joueur{i:02d}" for i in range(1, 16)]
    return pd.DataFrame({
        "player_name":        [players[i % 15] for i in range(n)],
        "game_id":            ["shooter"] * n,
        "score":              rng.integers(100, 900, n),
        "duration_sec":       rng.integers(30, 300, n),
        "btn_press_rate":     rng.uniform(0.1, 1.0, n),
        "btn_variety":        rng.uniform(0.1, 1.0, n),
        "btn_hold_avg_ms":    rng.uniform(50, 400, n),
        "lx_mean":            rng.uniform(-0.5, 0.5, n),
        "ly_mean":            rng.uniform(-0.5, 0.5, n),
        "lx_std":             rng.uniform(0.1, 0.8, n),
        "ly_std":             rng.uniform(0.1, 0.8, n),
        "lx_direction_changes": rng.integers(5, 50, n).astype(float),
        "rx_std":             rng.uniform(0.1, 0.8, n),
        "ry_std":             rng.uniform(0.1, 0.8, n),
        "rt_mean":            rng.uniform(0.0, 0.8, n),
        "lt_mean":            rng.uniform(0.0, 0.5, n),
        "input_regularity":   rng.uniform(0.2, 0.9, n),
        "created_at":         pd.date_range("2025-01-01", periods=n, freq="2H"),
    })


def main():
    print("=" * 60)
    print("  SISE Gaming — Analyse Comportementale Shooter")
    print("=" * 60)

    df = _load_data()
    result = compute_shooter_analysis(df)

    if result is None:
        print("❌ Données shooter insuffisantes pour l'analyse.")
        return

    cl = result["clustering"]
    pr = result["progression"]
    co = result["correlation"]

    # ── Résumé console ────────────────────────────────────────────────────────
    print(f"\n📊 DONNÉES  : {cl['n']} sessions shooter")
    print(f"\n🔵 CLUSTERING")
    print(f"   Silhouette scores : {cl['silhouette_scores']}")
    print(f"   Meilleur k : {cl['best_k']}  (k=3 utilisé pour la suite)")
    print(f"   Clusters :")
    for c, name in cl["cluster_names"].items():
        n_c = (cl["labels"] == c).sum()
        print(f"     {c}: {name}  ({n_c} sessions)")

    print(f"\n📈 PROGRESSION")
    for player, status, slope, p in zip(pr["players"], pr["statuses"], pr["slopes"], pr["pvalues"]):
        print(f"   {player:20s} {status}  pente={slope:+.2f}  p={p:.3f}")

    print(f"\n🔗 CORRÉLATION → SCORE")
    for feat, r, p in zip(co["features"][:5], co["spearman_r"][:5], co["pvalues"][:5]):
        label = FEATURE_LABELS.get(feat, feat)
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        print(f"   {label:30s} r={r:+.3f}  p={p:.4f} {sig}")
    print(f"   → Top 3 prédictifs : {[FEATURE_LABELS.get(f, f) for f in co['top3']]}")

    # ── Sauvegarde optionnelle ────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "outputs")
    os.makedirs(out_dir, exist_ok=True)
    save = input("\n💾 Sauvegarder les figures ? (o/N) : ").strip().lower() == "o"
    sp_cl  = os.path.join(out_dir, "clustering.png")  if save else None
    sp_pr  = os.path.join(out_dir, "progression.png") if save else None
    sp_co  = os.path.join(out_dir, "correlation.png") if save else None

    print("\n🖼️  Affichage Figure 1 — Clustering…")
    plot_clustering(result, save_path=sp_cl)

    print("🖼️  Affichage Figure 2 — Progression…")
    plot_progression(result, save_path=sp_pr)

    print("🖼️  Affichage Figure 3 — Corrélation…")
    plot_correlation(result, save_path=sp_co)

    print("\n✅ Analyse terminée.")


if __name__ == "__main__":
    main()
