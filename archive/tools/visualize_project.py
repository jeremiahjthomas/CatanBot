"""
visualize_project.py

Generates a project architecture overview PNG showing:
- Branch timeline / git history
- Module dependency graph
- Component summary (files, lines, tests)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.patheffects as pe
import numpy as np

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

BRANCHES = [
    {
        "name": "visualization",
        "color": "#4e9af1",
        "desc": "Board Visualization",
        "files": ["env/board.py", "visualize_board.py"],
        "tests": 0,
        "lines": 280,
        "highlights": [
            "19-hex pointy-top grid",
            "54 vertices, 72 edges",
            "9 coastal port slots",
            "Port-to-vertex lines",
            "Edge/vertex labels",
        ],
    },
    {
        "name": "balancedDie",
        "color": "#f4a261",
        "desc": "Balanced Dice Engine",
        "files": ["env/balanced_dice.py", "visualize_dice.py", "test_balanced_dice.py"],
        "tests": 24,
        "lines": 310,
        "highlights": [
            "Port of Colonist.io algorithm",
            "Deck of 36 (d1,d2) pairs",
            "Recent-roll penalty (0.34x)",
            "Seven streak/imbalance adj.",
            "Dice distribution visualizer",
        ],
    },
    {
        "name": "gameState",
        "color": "#2ec4b6",
        "desc": "Game State",
        "files": ["env/game_state.py", "test_game_state.py"],
        "tests": 57,
        "lines": 620,
        "highlights": [
            "Resource, DevCard, GamePhase enums",
            "PlayerState + GameState dataclasses",
            "new_game() initialization",
            "Legal move query functions",
            "Longest road DFS",
        ],
    },
    {
        "name": "environment",
        "color": "#e76f51",
        "desc": "RL Environment",
        "files": ["env/actions.py", "env/catan_env.py", "test_catan_env.py"],
        "tests": 62,
        "lines": 810,
        "highlights": [
            "249-action flat discrete space",
            "Phase-aware action mask",
            "All apply_* state transitions",
            "460-dim observation vector",
            "Gym-compatible CatanEnv",
        ],
    },
]

MODULES = {
    "env/board.py":        (0.18, 0.72),
    "env/balanced_dice.py": (0.18, 0.50),
    "env/game_state.py":   (0.50, 0.72),
    "env/actions.py":      (0.50, 0.50),
    "env/catan_env.py":    (0.50, 0.28),
    "visualize_board.py":  (0.82, 0.72),
    "visualize_dice.py":   (0.82, 0.50),
    "CatanEnv\n(Gym API)": (0.82, 0.28),
}

MODULE_COLORS = {
    "env/board.py":         "#4e9af1",
    "env/balanced_dice.py": "#f4a261",
    "env/game_state.py":    "#2ec4b6",
    "env/actions.py":       "#e76f51",
    "env/catan_env.py":     "#e76f51",
    "visualize_board.py":   "#4e9af1",
    "visualize_dice.py":    "#f4a261",
    "CatanEnv\n(Gym API)":  "#c77dff",
}

DEPS = [
    ("env/board.py",         "env/game_state.py"),
    ("env/board.py",         "visualize_board.py"),
    ("env/balanced_dice.py", "visualize_dice.py"),
    ("env/balanced_dice.py", "env/catan_env.py"),
    ("env/game_state.py",    "env/actions.py"),
    ("env/game_state.py",    "env/catan_env.py"),
    ("env/actions.py",       "env/catan_env.py"),
    ("env/catan_env.py",     "CatanEnv\n(Gym API)"),
]

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(18, 12), facecolor="#0d1117")
fig.suptitle("CatanBot — Project Architecture", fontsize=22, fontweight="bold",
             color="white", y=0.97)

# Three sections: branch timeline (top), module graph (middle-left), stats (middle-right)
gs = fig.add_gridspec(
    3, 2,
    left=0.03, right=0.97,
    top=0.92, bottom=0.03,
    hspace=0.35, wspace=0.25,
    height_ratios=[1.1, 1.6, 0.9],
)

ax_timeline = fig.add_subplot(gs[0, :])
ax_deps     = fig.add_subplot(gs[1, 0])
ax_stats    = fig.add_subplot(gs[1, 1])
ax_detail   = fig.add_subplot(gs[2, :])

for ax in [ax_timeline, ax_deps, ax_stats, ax_detail]:
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

# ---------------------------------------------------------------------------
# 1. Branch timeline
# ---------------------------------------------------------------------------

ax_timeline.set_xlim(-0.5, len(BRANCHES) - 0.5)
ax_timeline.set_ylim(-0.3, 1.2)
ax_timeline.axis("off")
ax_timeline.set_title("Branch Timeline  (main ← merges)", color="#8b949e",
                       fontsize=12, loc="left", pad=6)

# main line
ax_timeline.axhline(0.5, color="#30363d", lw=2, zorder=0)
# origin dot
ax_timeline.plot(-0.5, 0.5, "o", color="#8b949e", ms=8, zorder=3)
ax_timeline.text(-0.5, 0.2, "main", color="#8b949e", ha="center", fontsize=9)

for i, br in enumerate(BRANCHES):
    x = i
    c = br["color"]
    # branch line up then across then down
    ax_timeline.plot([x - 0.35, x - 0.35, x + 0.35, x + 0.35],
                     [0.5, 0.85, 0.85, 0.5],
                     color=c, lw=2, zorder=1)
    # merge arrow back to main
    ax_timeline.annotate("", xy=(x + 0.35, 0.5), xytext=(x + 0.35, 0.57),
                         arrowprops=dict(arrowstyle="-|>", color=c, lw=1.5))
    # branch name bubble
    bbox = dict(boxstyle="round,pad=0.4", fc=c, ec="none", alpha=0.85)
    ax_timeline.text(x, 1.05, br["name"], ha="center", va="center",
                     color="white", fontsize=10, fontweight="bold", bbox=bbox)
    # desc below
    ax_timeline.text(x, 0.83, br["desc"], ha="center", va="bottom",
                     color="#c9d1d9", fontsize=8.5)
    # test count
    if br["tests"]:
        ax_timeline.text(x, 0.55, f'{br["tests"]} tests', ha="center", va="bottom",
                         color=c, fontsize=8, fontweight="bold")

# ---------------------------------------------------------------------------
# 2. Module dependency graph
# ---------------------------------------------------------------------------

ax_deps.set_xlim(0, 1)
ax_deps.set_ylim(0, 1)
ax_deps.axis("off")
ax_deps.set_title("Module Dependencies", color="#8b949e", fontsize=12, loc="left", pad=6)

NODE_W, NODE_H = 0.22, 0.09

def node_center(name):
    return MODULES[name]

# Draw edges first
for src, dst in DEPS:
    sx, sy = node_center(src)
    dx, dy = node_center(dst)
    # offset so arrows start/end at node edges
    ax_deps.annotate(
        "", xy=(dx, dy), xytext=(sx, sy),
        xycoords="data", textcoords="data",
        arrowprops=dict(
            arrowstyle="-|>",
            color="#444c56",
            lw=1.5,
            connectionstyle="arc3,rad=0.08",
        ),
    )

# Draw nodes
for name, (x, y) in MODULES.items():
    c = MODULE_COLORS[name]
    box = FancyBboxPatch(
        (x - NODE_W / 2, y - NODE_H / 2), NODE_W, NODE_H,
        boxstyle="round,pad=0.01",
        facecolor=c + "33",
        edgecolor=c,
        linewidth=1.8,
        transform=ax_deps.transData,
        clip_on=False,
    )
    ax_deps.add_patch(box)
    short = name.split("/")[-1]
    ax_deps.text(x, y, short, ha="center", va="center",
                 color="white", fontsize=8, fontweight="bold",
                 multialignment="center")

# legend
for label, color in [("visualization", "#4e9af1"), ("balancedDie", "#f4a261"),
                      ("gameState", "#2ec4b6"), ("environment", "#e76f51"),
                      ("entry point", "#c77dff")]:
    pass  # embedded in colors already

# ---------------------------------------------------------------------------
# 3. Stats bars
# ---------------------------------------------------------------------------

ax_stats.set_facecolor("#161b22")
ax_stats.set_title("Lines of Code  &  Test Count by Branch", color="#8b949e",
                    fontsize=12, loc="left", pad=6)

names   = [b["name"] for b in BRANCHES]
lines   = [b["lines"] for b in BRANCHES]
tests   = [b["tests"] for b in BRANCHES]
colors  = [b["color"] for b in BRANCHES]
x_pos   = np.arange(len(BRANCHES))
bar_w   = 0.35

bars_l = ax_stats.bar(x_pos - bar_w / 2, lines, bar_w, color=colors, alpha=0.75, label="Lines")
bars_t = ax_stats.bar(x_pos + bar_w / 2, tests, bar_w, color=colors, alpha=0.45, label="Tests",
                       hatch="//", edgecolor="white", linewidth=0.5)

ax_stats.set_xticks(x_pos)
ax_stats.set_xticklabels(names, color="#c9d1d9", fontsize=9)
ax_stats.set_ylabel("Count", color="#8b949e", fontsize=9)
ax_stats.tick_params(colors="#8b949e")
ax_stats.set_facecolor("#161b22")
for spine in ax_stats.spines.values():
    spine.set_edgecolor("#30363d")
ax_stats.yaxis.label.set_color("#8b949e")
ax_stats.tick_params(axis="y", colors="#8b949e")

# value labels
for bar in bars_l:
    h = bar.get_height()
    ax_stats.text(bar.get_x() + bar.get_width() / 2, h + 8, str(int(h)),
                  ha="center", va="bottom", color="#c9d1d9", fontsize=8)
for bar in bars_t:
    h = bar.get_height()
    if h > 0:
        ax_stats.text(bar.get_x() + bar.get_width() / 2, h + 8, str(int(h)),
                      ha="center", va="bottom", color="#c9d1d9", fontsize=8)

solid_patch = mpatches.Patch(color="#8b949e", alpha=0.75, label="Lines of code")
hatch_patch = mpatches.Patch(facecolor="#8b949e", alpha=0.45, hatch="//",
                              edgecolor="white", label="Test count")
ax_stats.legend(handles=[solid_patch, hatch_patch], facecolor="#0d1117",
                edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

# ---------------------------------------------------------------------------
# 4. Feature detail table
# ---------------------------------------------------------------------------

ax_detail.axis("off")
ax_detail.set_title("Branch Highlights", color="#8b949e", fontsize=12, loc="left", pad=6)

col_w = 1.0 / len(BRANCHES)
for i, br in enumerate(BRANCHES):
    cx = (i + 0.5) * col_w
    # header
    ax_detail.text(cx, 0.95, br["name"], ha="center", va="top",
                   color=br["color"], fontsize=10, fontweight="bold",
                   transform=ax_detail.transAxes)
    ax_detail.text(cx, 0.80, br["desc"], ha="center", va="top",
                   color="#8b949e", fontsize=8, transform=ax_detail.transAxes)
    # bullet points
    for j, hl in enumerate(br["highlights"]):
        y = 0.66 - j * 0.15
        ax_detail.text(cx - col_w * 0.44, y, f"• {hl}",
                       ha="left", va="top", color="#c9d1d9", fontsize=8,
                       transform=ax_detail.transAxes)
    # vertical divider
    if i < len(BRANCHES) - 1:
        ax_detail.plot([(i + 1) * col_w, (i + 1) * col_w], [0, 1],
                       color="#30363d", lw=1, transform=ax_detail.transAxes,
                       clip_on=False)

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

plt.savefig("project_overview.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved -> project_overview.png")
