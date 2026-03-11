"""
visualize_dice.py  —  visualise how the balanced dice engine behaves

Two figures:

  Figure 1 — Recent-roll penalty (applies to ALL totals, not just 7)
    1. Fresh engine (baseline)
    2. 8 rolled twice (penalty on one common number)
    3. 6 and 8 each rolled once (two common numbers hit)
    4. 5, 6, 8, 9 each rolled once (four numbers penalised, memory nearly full)
    5. 7 rolled 5 times (penalty pushes 7 to 0)
    6. Realistic game: rolls were 8, 5, 6, 9, 8 (8 appears twice, others once)

  Figure 2 — Seven-specific adjustments (imbalance + streak)
    1. Fresh engine
    2. Imbalance: P0 over-represented in 7s — P0 view
    3. Imbalance: P0 over-represented in 7s — P1 view
    4. Streak: P0 on 3-roll 7-streak — P0 view
    5. Streak: P0 on 3-roll 7-streak — P1 view
    6. Combined: P0 over-represented AND on streak — P1 view (double boost)

Usage:
  python visualize_dice.py           # opens both figures
  python visualize_dice.py save      # saves to dice_viz_recent.png + dice_viz_sevens.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from env.balanced_dice import (
    BalancedDiceEngine,
    MAX_RECENT_MEMORY,
    _STANDARD_DECK,
)

SAVE = "save" in sys.argv

TOTALS = list(range(2, 13))
STANDARD_PROBS = {t: len(_STANDARD_DECK[t]) / 36 for t in TOTALS}

COL_STANDARD = "#9E9E9E"
COL_DIST     = "#1565C0"
COL_REDUCED  = "#C62828"
COL_BOOSTED  = "#2E7D32"


def bar_color(dist_p: float, std_p: float) -> str:
    delta = dist_p - std_p
    if abs(delta) < 0.004:
        return COL_DIST
    return COL_REDUCED if delta < 0 else COL_BOOSTED


def draw_panel(ax, title: str, dist: dict, note: str = "",
               recent: list | None = None) -> None:
    xs       = np.arange(len(TOTALS))
    std_vals = [STANDARD_PROBS[t] for t in TOTALS]
    dist_vals= [dist[t] for t in TOTALS]
    colors   = [bar_color(dist[t], STANDARD_PROBS[t]) for t in TOTALS]

    ax.bar(xs, std_vals,  width=0.6, color=COL_STANDARD, alpha=0.30, zorder=2)
    ax.bar(xs, dist_vals, width=0.4, color=colors,       alpha=0.88, zorder=3)

    for x, t in zip(xs, TOTALS):
        delta = dist[t] - STANDARD_PROBS[t]
        if abs(delta) > 0.003:
            sign = "+" if delta > 0 else ""
            col  = COL_BOOSTED if delta > 0 else COL_REDUCED
            ax.text(x, dist[t] + 0.003, f"{sign}{delta*100:.1f}%",
                    ha="center", va="bottom", fontsize=5.5, color=col,
                    fontweight="bold")

    # Mark totals that appear in the recent window
    if recent:
        from collections import Counter as _C
        rc = _C(recent)
        for x, t in zip(xs, TOTALS):
            if t in rc:
                ax.text(x, -0.013, f"×{rc[t]}", ha="center", va="top",
                        fontsize=6, color="#880000", fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels(TOTALS, fontsize=8)
    ax.set_ylim(-0.018, 0.25)
    ax.set_ylabel("Probability", fontsize=7)
    ax.set_title(title, fontsize=8.5, fontweight="bold", pad=5)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}" if v >= 0 else ""))
    ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.spines[["top", "right"]].set_visible(False)

    if note:
        ax.text(0.98, 0.98, note, transform=ax.transAxes,
                fontsize=6.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFF9C4",
                          edgecolor="#F9A825", linewidth=0.7))


def make_engine(recent: list, sevens: dict | None = None,
                streak_player: int | None = None, streak_count: int = 0,
                seed: int = 0) -> BalancedDiceEngine:
    """Build an engine with manually injected state for demonstration."""
    from collections import Counter as _C
    e = BalancedDiceEngine(num_players=2, seed=seed)
    e._recent_rolls = list(recent)
    rc = _C(recent)
    e._recent_count = {t: rc.get(t, 0) for t in range(2, 13)}
    if sevens:
        e._total_sevens = dict(sevens)
    if streak_player is not None:
        e._streak_player = streak_player
        e._streak_count  = streak_count
    return e


# ---------------------------------------------------------------------------
# Figure 1 — Recent-roll penalty
# ---------------------------------------------------------------------------

scenarios_recent = [
    # (title, recent_rolls, note)
    ("1. Fresh Engine",
     [],
     "No rolls yet\nAll probabilities = standard 2d6"),

    ("2. Recent: [8, 8]\n(8 rolled twice)",
     [8, 8],
     "8 in window ×2\nweight[8] × (1−2×0.34) = ×0.32"),

    ("3. Recent: [6, 8]\n(one each of two common numbers)",
     [6, 8],
     "6 and 8 each in window ×1\nboth reduced by ×0.66"),

    ("4. Recent: [5, 6, 8, 9]\n(four numbers hit, window nearly full)",
     [5, 6, 8, 9],
     "4 of 5 recent slots used\nAll four reduced by ×0.66"),

    ("5. Recent: [7,7,7,7,7]\n(7 rolled 5 times — window full)",
     [7, 7, 7, 7, 7],
     "7 weight → 1−5×0.34 = −0.7\nclamped to 0"),

    ("6. Realistic: [8, 5, 6, 9, 8]\n(8 twice + three others)",
     [8, 5, 6, 9, 8],
     "8 penalised most (×2 in window)\n5, 6, 9 each penalised once"),
]

fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8))
fig1.suptitle("Balanced Dice: Recent-Roll Penalty\n"
              "(applies to ALL totals — red × labels show how many times a total is in the recent window)",
              fontsize=11, fontweight="bold")

for ax, (title, recent, note) in zip(axes1.flat, scenarios_recent):
    e = make_engine(recent)
    dist = e.get_distribution(player_id=0)
    draw_panel(ax, title, dist, note=note, recent=recent)

legend_handles = [
    mpatches.Patch(color=COL_STANDARD, alpha=0.45, label="Standard 2d6"),
    mpatches.Patch(color=COL_DIST,     alpha=0.88, label="Balanced (unchanged)"),
    mpatches.Patch(color=COL_REDUCED,  alpha=0.88, label="Balanced (reduced)"),
    mpatches.Patch(color=COL_BOOSTED,  alpha=0.88, label="Balanced (boosted)"),
]
fig1.legend(handles=legend_handles, loc="lower center", ncol=4,
            fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.01))
fig1.text(0.5, -0.04, "Red ×N below x-axis = total appeared N times in recent window",
          ha="center", fontsize=8, color="#880000")
plt.tight_layout()

# ---------------------------------------------------------------------------
# Figure 2 — Seven-specific adjustments
# ---------------------------------------------------------------------------

e_fresh  = make_engine([])
e_imb_p0 = make_engine([], sevens={0: 10, 1: 2})
e_imb_p1 = make_engine([], sevens={0: 10, 1: 2})
e_str_p0 = make_engine([], streak_player=0, streak_count=3)
e_str_p1 = make_engine([], streak_player=0, streak_count=3)
e_combo  = make_engine([], sevens={0: 10, 1: 2}, streak_player=0, streak_count=3)

scenarios_sevens = [
    ("1. Fresh Engine",
     e_fresh.get_distribution(0), 0,
     "No seven adjustments"),

    ("2. Imbalance — P0 view\n(P0: 10 sevens, P1: 2)",
     e_imb_p0.get_distribution(0), 0,
     "P0 over-represented\n7 reduced for P0"),

    ("3. Imbalance — P1 view\n(P0: 10 sevens, P1: 2)",
     e_imb_p1.get_distribution(1), 1,
     "P1 under-represented\n7 boosted for P1"),

    ("4. Streak — P0 view\n(P0 on 3-roll 7-streak)",
     e_str_p0.get_distribution(0), 0,
     "P0 is the streaker\n7 penalised for P0"),

    ("5. Streak — P1 view\n(P0 on 3-roll 7-streak)",
     e_str_p1.get_distribution(1), 1,
     "P1 is not the streaker\n7 boosted for P1"),

    ("6. Combined — P1 view\n(P0: 10 sevens + 3-streak)",
     e_combo.get_distribution(1), 1,
     "Both imbalance AND streak\nbenefit P1 → double boost on 7"),
]

fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
fig2.suptitle("Balanced Dice: Seven-Specific Adjustments\n"
              "(imbalance + streak — only 7 is affected by these)",
              fontsize=11, fontweight="bold")

for ax, (title, dist, pid, note) in zip(axes2.flat, scenarios_sevens):
    draw_panel(ax, title, dist, note=note)

fig2.legend(handles=legend_handles, loc="lower center", ncol=4,
            fontsize=8, frameon=True, bbox_to_anchor=(0.5, -0.01))
plt.tight_layout()

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

if SAVE:
    fig1.savefig("dice_viz_recent.png",  dpi=150, bbox_inches="tight")
    fig2.savefig("dice_viz_sevens.png",  dpi=150, bbox_inches="tight")
    print("Saved -> dice_viz_recent.png")
    print("Saved -> dice_viz_sevens.png")
else:
    plt.show()
