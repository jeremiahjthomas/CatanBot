"""
Catan board visualizer.

Usage:
    python visualize_board.py               # random board, opens window
    python visualize_board.py 42            # seed 42
    python visualize_board.py 42 save       # save to board_viz.png instead

Shows:
  - Hexes colored by resource type
  - Number tokens with pip dots (red for 6 & 8)
  - Vertex IDs (small grey text)
  - Robber position (bold red border)
  - Hex IDs (small text, top-left of each hex)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sys
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from env.board import generate_board, SQRT3, PORT_SLOT_DEFINITIONS, _EDGE_LIST

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

RESOURCE_COLORS = {
    "wood":   "#2d6e2d",
    "brick":  "#c0522a",
    "sheep":  "#7dba3e",
    "wheat":  "#e8b800",
    "ore":    "#7a8c99",
    "desert": "#d4c5a0",
}

RESOURCE_LABELS = {
    "wood":   "Wood (Forest)",
    "brick":  "Brick (Hills)",
    "sheep":  "Sheep (Pasture)",
    "wheat":  "Wheat (Fields)",
    "ore":    "Ore (Mountains)",
    "desert": "Desert",
}

# Number of pip dots per token value
PIP_COUNT = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

TOKEN_COLOR_HIGH = "#c0392b"   # red for 6 & 8
TOKEN_COLOR_NORM = "#1a1a1a"   # near-black for others

PORT_COLORS = {
    "wood":  "#2d6e2d",
    "brick": "#c0522a",
    "sheep": "#7dba3e",
    "wheat": "#e8b800",
    "ore":   "#7a8c99",
    "3:1":   "#f5a623",
}

PORT_LABELS = {
    "wood":  "2:1\nWood",
    "brick": "2:1\nBrick",
    "sheep": "2:1\nSheep",
    "wheat": "2:1\nWheat",
    "ore":   "2:1\nOre",
    "3:1":   "3:1\nPort",
}


def draw_board(board, show_vertex_ids=True, show_hex_ids=True, title="Catan Board"):
    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=12)

    size = 1.0

    for h in board.hexes:
        cx = size * (SQRT3 * h.q + SQRT3 / 2.0 * h.r)
        cy = size * 1.5 * h.r

        # Hex vertices (pointy-top, angles 30°,90°,…)
        verts = [
            (cx + size * math.cos(math.radians(30 + 60 * i)),
             cy + size * math.sin(math.radians(30 + 60 * i)))
            for i in range(6)
        ]

        is_robber = h.hex_id == board.robber_hex
        facecolor = RESOURCE_COLORS[h.resource]
        edgecolor = "#8B0000" if is_robber else "#444444"
        linewidth = 3.5 if is_robber else 1.5

        poly = Polygon(verts, closed=True,
                       facecolor=facecolor, edgecolor=edgecolor,
                       linewidth=linewidth, zorder=1)
        ax.add_patch(poly)

        # Hex ID (small, upper-left corner of hex)
        if show_hex_ids:
            ax.text(cx - size * 0.52, cy + size * 0.62,
                    str(h.hex_id),
                    fontsize=6, color="#555555", zorder=3,
                    ha="center", va="center")

        if h.number_token > 0:
            num_color = TOKEN_COLOR_HIGH if h.number_token in (6, 8) else TOKEN_COLOR_NORM

            # Token circle background
            circle = plt.Circle((cx, cy + 0.04), 0.31,
                                 color="#fffde7", ec="#aaaaaa",
                                 linewidth=1, zorder=2)
            ax.add_patch(circle)

            # Number
            ax.text(cx, cy + 0.1, str(h.number_token),
                    fontsize=10, fontweight="bold", color=num_color,
                    ha="center", va="center", zorder=3)

            # Pip dots below number
            pips = PIP_COUNT.get(h.number_token, 0)
            pip_y = cy - 0.1
            pip_spacing = 0.09
            for p in range(pips):
                px_ = cx + (p - (pips - 1) / 2.0) * pip_spacing
                dot = plt.Circle((px_, pip_y), 0.027,
                                  color=num_color, zorder=3)
                ax.add_patch(dot)

        elif h.resource == "desert":
            # Robber icon
            ax.text(cx, cy, "R",
                    fontsize=16, fontweight="bold", color="#555555",
                    ha="center", va="center", zorder=3)

    # Ports — draw before vertices so vertices render on top
    for eid in PORT_SLOT_DEFINITIONS:
        va_id, vb_id = _EDGE_LIST[eid]
        ax_v = board.vertices[va_id]
        bx_v = board.vertices[vb_id]
        ptype = ax_v.port  # both vertices share the same port type

        if ptype is None:
            continue

        color = PORT_COLORS.get(ptype, "#f5a623")
        label = PORT_LABELS.get(ptype, ptype)

        # Midpoint of the port edge
        mx = (ax_v.x + bx_v.x) / 2
        my = (ax_v.y + bx_v.y) / 2

        # Push the label outward from board center (0,0)
        dist = math.hypot(mx, my)
        if dist > 0:
            nx, ny = mx / dist, my / dist  # outward unit vector
        else:
            nx, ny = 0, 1
        label_x = mx + nx * 0.55
        label_y = my + ny * 0.55

        # Lines from label to each vertex
        ax.plot([label_x, ax_v.x], [label_y, ax_v.y],
                color=color, linewidth=2.5, solid_capstyle="round", zorder=3)
        ax.plot([label_x, bx_v.x], [label_y, bx_v.y],
                color=color, linewidth=2.5, solid_capstyle="round", zorder=3)

        # Port label box
        ax.text(label_x, label_y, label,
                fontsize=9, fontweight="bold", color="#ffffff",
                ha="center", va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.35", facecolor=color,
                          edgecolor="white", linewidth=1.2, alpha=0.95),
                multialignment="center")

    # Coastal edge labels
    from env.board import _EDGE_TO_HEXES
    for eid, (va_id, vb_id) in enumerate(_EDGE_LIST):
        if len(_EDGE_TO_HEXES[eid]) == 1:
            va_v = board.vertices[va_id]
            vb_v = board.vertices[vb_id]
            mx = (va_v.x + vb_v.x) / 2
            my = (va_v.y + vb_v.y) / 2
            # Push label inward slightly
            dist = math.hypot(mx, my)
            if dist > 0:
                inx, iny = mx / dist, my / dist
            else:
                inx, iny = 0, 1
            lx = mx - inx * 0.28
            ly = my - iny * 0.28
            ax.text(lx, ly, str(eid),
                    fontsize=5.5, color="#cc3300", fontweight="bold",
                    ha="center", va="center", zorder=7,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                              edgecolor="#cc3300", linewidth=0.6, alpha=0.85))

    # Vertices
    for v in board.vertices:
        color = PORT_COLORS.get(v.port, "#444444") if v.port else "#444444"
        size_pt = 5.5 if v.port else 3.5
        ax.plot(v.x, v.y, "o", markersize=size_pt, color=color,
                markeredgecolor="#ffffff", markeredgewidth=0.7, zorder=4)
        if show_vertex_ids:
            ax.text(v.x + 0.06, v.y + 0.06, str(v.vertex_id),
                    fontsize=4.5, color="#888888", zorder=5)

    # Legend — terrain tiles
    terrain_patches = [
        mpatches.Patch(color=RESOURCE_COLORS[r], label=RESOURCE_LABELS[r])
        for r in ["wood", "brick", "sheep", "wheat", "ore", "desert"]
    ]
    # Port type patches
    port_patches = [
        mpatches.Patch(color=PORT_COLORS["3:1"],   label="3:1 Port (generic)"),
        mpatches.Patch(color=PORT_COLORS["wood"],  label="2:1 Wood port"),
        mpatches.Patch(color=PORT_COLORS["brick"], label="2:1 Brick port"),
        mpatches.Patch(color=PORT_COLORS["sheep"], label="2:1 Sheep port"),
        mpatches.Patch(color=PORT_COLORS["wheat"], label="2:1 Wheat port"),
        mpatches.Patch(color=PORT_COLORS["ore"],   label="2:1 Ore port"),
    ]
    ax.legend(handles=terrain_patches + port_patches,
              loc="lower right", fontsize=7.5,
              framealpha=0.88, edgecolor="#cccccc",
              title="Terrain & Ports", title_fontsize=8)

    # Stats box
    resource_counts = {}
    for h in board.hexes:
        resource_counts[h.resource] = resource_counts.get(h.resource, 0) + 1

    stats = (
        f"Hexes: {len(board.hexes)}  |  "
        f"Vertices: {len(board.vertices)}  |  "
        f"Edges: {len(board.edges)}\n"
        f"Robber: hex {board.robber_hex} ({board.hexes[board.robber_hex].resource})"
    )
    ax.text(0.01, 0.01, stats,
            transform=ax.transAxes, fontsize=7.5,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f8f8",
                      edgecolor="#cccccc", alpha=0.85))

    plt.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    seed_arg = sys.argv[1] if len(sys.argv) > 1 else None
    save_arg  = "save" in sys.argv

    seed = int(seed_arg) if seed_arg and seed_arg != "save" else None
    board = generate_board(seed=seed)

    print(f"Board generated  seed={seed}")
    print(f"  hexes={len(board.hexes)}  vertices={len(board.vertices)}  edges={len(board.edges)}")
    print(f"  robber on hex {board.robber_hex} ({board.hexes[board.robber_hex].resource})")

    title = f"Catan Board  (seed={seed})" if seed is not None else "Catan Board  (random)"
    draw_board(board, show_vertex_ids=True, title=title)

    if save_arg:
        out = "board_viz.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved -> {out}")
    else:
        plt.show()
