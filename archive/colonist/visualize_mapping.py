"""
visualize_mapping.py

Generates two board diagrams side by side:

  LEFT:  CatanEnv board with all vertex IDs (0–53) and edge IDs (0–71).
         This is the reference — you can see exactly which number maps to
         which geometric position on the board.

  RIGHT: Same board geometry but blank labels.  Buildings and roads from
         the captured Colonist game are drawn with *Colonist* IDs shown.
         Compare the position of each Colonist piece to the left panel
         to read off the CatanEnv ID.

Usage:
    python visualize_mapping.py <game_id>           # open interactive window
    python visualize_mapping.py <game_id> save      # save <game_id>_mapping.png
"""

import sys
import math
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon

from env.board import (
    generate_board, SQRT3,
    _VERTEX_POSITIONS, _EDGE_LIST,
    CATAN_HEX_AXIAL, _EDGE_TO_HEXES,
)

# ── colours ────────────────────────────────────────────────────────────────
PLAYER_COLORS = {1: "#e74c3c", 5: "#2575c4"}   # red / blue
SIZE = 1.0

# ── helpers ────────────────────────────────────────────────────────────────
def hex_verts(q, r):
    cx = SIZE * (SQRT3 * q + SQRT3 / 2 * r)
    cy = SIZE * 1.5 * r
    return [(cx + SIZE * math.cos(math.radians(30 + 60*i)),
             cy + SIZE * math.sin(math.radians(30 + 60*i)))
            for i in range(6)]

def edge_mid(eid):
    va, vb = _EDGE_LIST[eid]
    xa, ya = _VERTEX_POSITIONS[va]
    xb, yb = _VERTEX_POSITIONS[vb]
    return (xa+xb)/2, (ya+yb)/2

# ── load colonist state ─────────────────────────────────────────────────────
def load_colonist(game_id):
    p = Path(__file__).parent / "replays" / f"replay_{game_id}_gamedata.json"
    if not p.exists():
        print(f"Game data not found: {p}")
        return {}, {}
    events = json.loads(p.read_text())["data"]["eventHistory"]["events"]
    corners, edges = {}, {}
    for ev in events:
        ms = ev.get("stateChange", {}).get("mapState", {})
        for k, v in ms.get("tileCornerStates", {}).items():
            corners.setdefault(int(k), {}).update(v)
        for k, v in ms.get("tileEdgeStates", {}).items():
            edges.setdefault(int(k), {}).update(v)
    return corners, edges

# ── draw one hex grid ───────────────────────────────────────────────────────
def draw_grid(ax, board,
              vertex_labels,   # dict vid -> label string (or None to hide)
              edge_labels,     # dict eid -> label string (or None to hide)
              colonist_corners=None,  # Colonist vid -> {owner, buildingType}
              colonist_edges=None,    # Colonist eid -> {owner}
              title=""):

    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=10, fontweight="bold", pad=6)

    # hexes — blank fill, no resources or numbers (they'd be wrong anyway)
    for h in board.hexes:
        pts = hex_verts(h.q, h.r)
        ax.add_patch(Polygon(pts, closed=True,
                             facecolor="#e8e4d8",
                             edgecolor="#888", linewidth=1.1, zorder=1))

    # all edges (thin grey first, then coloured roads on top)
    for eid, (va, vb) in enumerate(_EDGE_LIST):
        xa, ya = _VERTEX_POSITIONS[va]
        xb, yb = _VERTEX_POSITIONS[vb]
        ax.plot([xa,xb],[ya,yb], color="#d0d0d0",
                linewidth=0.9, solid_capstyle="round", zorder=2)

    # colonist roads
    if colonist_edges:
        for c_eid, ed in colonist_edges.items():
            owner = ed.get("owner")
            col = PLAYER_COLORS.get(owner, "#888")
            if c_eid < len(_EDGE_LIST):
                va, vb = _EDGE_LIST[c_eid]
                xa, ya = _VERTEX_POSITIONS[va]
                xb, yb = _VERTEX_POSITIONS[vb]
                ax.plot([xa,xb],[ya,yb], color=col,
                        linewidth=5, solid_capstyle="round",
                        zorder=3, alpha=0.7)

    # edge labels
    for eid in range(len(_EDGE_LIST)):
        lbl = edge_labels.get(eid)
        if lbl is None:
            continue
        mx, my = edge_mid(eid)
        road_col = None
        if colonist_edges and eid in colonist_edges:
            road_col = PLAYER_COLORS.get(colonist_edges[eid].get("owner"))
        fc = road_col if road_col else "white"
        tc = "white" if road_col else "#cc4400"
        ec = "white" if road_col else "#cc4400"
        ax.text(mx, my, lbl, fontsize=6, color=tc,
                ha="center", va="center", zorder=7, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", facecolor=fc,
                          edgecolor=ec, linewidth=0.6, alpha=0.9))

    # all vertices (small dot)
    for vid, (vx, vy) in enumerate(_VERTEX_POSITIONS):
        ax.plot(vx, vy, "o", markersize=2.5, color="#666",
                markeredgecolor="#fff", markeredgewidth=0.4, zorder=4)

    # colonist buildings (on top of dots)
    if colonist_corners:
        for c_vid, cd in colonist_corners.items():
            owner = cd.get("owner")
            btype = cd.get("buildingType", 1)
            col = PLAYER_COLORS.get(owner, "#888")
            if c_vid < len(_VERTEX_POSITIONS):
                vx, vy = _VERTEX_POSITIONS[c_vid]
                mk = "s" if btype == 2 else "^"
                ms = 13 if btype == 2 else 10
                ax.plot(vx, vy, mk, markersize=ms, color=col,
                        markeredgecolor="white", markeredgewidth=1.1, zorder=6)

    # vertex labels
    for vid, (vx, vy) in enumerate(_VERTEX_POSITIONS):
        lbl = vertex_labels.get(vid)
        if lbl is None:
            continue
        # check if a building is here
        bld_col = None
        if colonist_corners and vid in colonist_corners:
            bld_col = PLAYER_COLORS.get(colonist_corners[vid].get("owner"))
        fc = bld_col if bld_col else "white"
        tc = "white" if bld_col else "#333"
        ec = bld_col if bld_col else "#999"
        ax.text(vx, vy, lbl, fontsize=7, color=tc, fontweight="bold",
                ha="center", va="center", zorder=8,
                bbox=dict(boxstyle="round,pad=0.12", facecolor=fc,
                          edgecolor=ec, linewidth=0.7, alpha=0.95))


# ── main ───────────────────────────────────────────────────────────────────
def main():
    args = [a for a in sys.argv[1:] if a != "save"]
    if not args:
        print("Usage: python visualize_mapping.py <game_id> [save]")
        sys.exit(1)
    game_id = args[0]
    save = "save" in sys.argv

    board = generate_board(seed=42)   # only used for topology; resources/numbers not shown
    colonist_corners, colonist_edges = load_colonist(game_id)

    # Which Colonist vertices/edges appear in the game
    c_vids = set(colonist_corners.keys())
    c_eids = set(colonist_edges.keys())

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(28, 14))
    fig.patch.set_facecolor("#f4f4f4")

    # ── LEFT: CatanEnv reference ───────────────────────────────────────────
    all_v = {v: str(v) for v in range(len(_VERTEX_POSITIONS))}
    all_e = {e: str(e) for e in range(len(_EDGE_LIST))}

    draw_grid(ax_l, board,
              vertex_labels=all_v,
              edge_labels=all_e,
              title="CatanEnv topology  •  vertex IDs 0–53 (white boxes)  •  edge IDs 0–71 (orange boxes)")

    # ── RIGHT: Colonist game — pieces labeled with Colonist IDs ────────────
    c_v_labels = {v: f"C{v}" for v in c_vids if v < len(_VERTEX_POSITIONS)}
    c_e_labels = {e: f"C{e}" for e in c_eids if e < len(_EDGE_LIST)}

    draw_grid(ax_r, board,
              vertex_labels=c_v_labels,
              edge_labels=c_e_labels,
              colonist_corners=colonist_corners,
              colonist_edges=colonist_edges,
              title=f"Colonist game {game_id}  •  pieces labeled with Colonist IDs (C##)\n"
                    "Piece positions are tentative — compare with left panel to derive mapping")

    # legend
    leg = [
        mpatches.Patch(color=PLAYER_COLORS[5], label="Player 5 (blue)"),
        mpatches.Patch(color=PLAYER_COLORS[1], label="Player 1 (red)"),
        mpatches.Patch(color="white", ec="#333",
                       label="▲ settlement   ■ city   — road"),
    ]
    fig.legend(handles=leg, loc="lower center", ncol=3,
               fontsize=10, framealpha=0.9, edgecolor="#bbb")

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if save:
        out = str(Path(__file__).parent / "replays" / f"{game_id}_mapping.png")
        plt.savefig(out, dpi=220, bbox_inches="tight")
        print(f"Saved -> {out}  (open and zoom in to read IDs)")
    else:
        plt.show()


if __name__ == "__main__":
    main()
