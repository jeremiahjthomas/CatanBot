"""
visualize_board.py

Renders the board from a captured colonist.io replay,
showing hex resources, dice pip numbers, and ports.

Usage:
    python colonist/visualize_board.py 214191014
    python colonist/visualize_board.py 214191014 save
"""
import sys
import math
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyArrowPatch

from env.board import _VERTEX_POSITIONS

# ── Vertex label lookup ──────────────────────────────────────────────────────
# CatanEnv uses cy = 1.5*r; colonist (after y-flip) uses cy = 1.5*(-y).
# Since colonist x == CatanEnv q and colonist y == CatanEnv r, vertex positions
# are (vx, -vy) in colonist's rendered space.
_MAPPING_FILE = Path(__file__).parent / "colonist_to_catan_mapping.json"
_mapping = json.loads(_MAPPING_FILE.read_text())
# catan_vid -> col_vid  (inverted from col_to_catan_vertex)
_CATAN_TO_COL = {v: int(k) for k, v in _mapping["col_to_catan_vertex"].items()}
# colonist-space vertex positions: (vx, -vy) for each catan_vid
_COL_VERTEX_POS: dict[int, tuple] = {}   # col_vid -> (x, y) in colonist render space
for catan_vid, (vx, vy) in enumerate(_VERTEX_POSITIONS):
    col_vid = _CATAN_TO_COL.get(catan_vid)
    if col_vid is not None:
        _COL_VERTEX_POS[col_vid] = (vx, -vy)

# ── Resource type mapping (Colonist type int -> name) ────────────────────────
RESOURCE_NAMES = {0: "desert", 1: "wood", 2: "brick", 3: "sheep", 4: "wheat", 5: "ore"}
RESOURCE_COLORS = {
    "desert": "#e0d090",
    "wood":   "#5d8a3c",
    "brick":  "#c0522a",
    "sheep":  "#90c040",
    "wheat":  "#e8c840",
    "ore":    "#909090",
}
PORT_NAMES = {1: "3:1", 2: "wood", 3: "brick", 4: "sheep", 5: "wheat", 6: "ore"}
PORT_COLORS = {
    1: "#dddddd",   # 3:1 any
    2: "#5d8a3c",   # wood
    3: "#c0522a",   # brick
    4: "#90c040",   # sheep
    5: "#e8c840",   # wheat
    6: "#909090",   # ore
}
HOT_NUMBERS = {6, 8}

SIZE = 1.0
SQRT3 = math.sqrt(3)


def hex_center(x, y):
    """Colonist uses screen coords (y down), flip y for matplotlib (y up)."""
    cx = SIZE * (SQRT3 * x + SQRT3 / 2 * y)
    cy = SIZE * 1.5 * (-y)
    return cx, cy


def hex_corners(x, y):
    cx, cy = hex_center(x, y)
    return [(cx + SIZE * math.cos(math.radians(30 + 60 * i)),
             cy + SIZE * math.sin(math.radians(30 + 60 * i)))
            for i in range(6)]


def edge_endpoints(x, y, z):
    """
    Each hex edge z connects two corners:
      z=0 (SW): corner 4 <-> corner 5
      z=1 (W):  corner 3 <-> corner 4
      z=2 (NW): corner 2 <-> corner 3
    (using flat-top hex, corners 0=E going CCW)
    """
    corners = hex_corners(x, y)
    # pointy-top: corner 0 = top-right (30°)
    z_to_corners = {0: (1, 2), 1: (2, 3), 2: (3, 4)}
    i, j = z_to_corners[z % 3]
    return corners[i], corners[j]


def draw_board(game_id, save=False):
    f = Path(__file__).parent / "replays" / f"replay_{game_id}_gamedata.json"
    if not f.exists():
        print(f"File not found: {f}")
        sys.exit(1)

    data = json.loads(f.read_text())["data"]["eventHistory"]["initialState"]["mapState"]
    tiles = data["tileHexStates"]
    ports = data.get("portEdgeStates", {})

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Game {game_id} — board layout", fontsize=14, pad=10)

    # ── Draw hex tiles ───────────────────────────────────────────────────────
    for tile in tiles.values():
        x, y = tile["x"], tile["y"]
        rtype = RESOURCE_NAMES.get(tile["type"], "desert")
        color = RESOURCE_COLORS[rtype]
        corners = hex_corners(x, y)
        poly = Polygon(corners, closed=True, facecolor=color,
                       edgecolor="#555", linewidth=1.2, zorder=1)
        ax.add_patch(poly)

        cx, cy = hex_center(x, y)

        # Resource label
        ax.text(cx, cy + 0.22, rtype.upper(), ha="center", va="center",
                fontsize=6.5, color="white" if rtype in ("wood", "ore", "brick") else "#333",
                fontweight="bold", zorder=3)

        # Dice number
        num = tile["diceNumber"]
        if num > 0:
            hot = num in HOT_NUMBERS
            fc = "#cc2200" if hot else "#fffbe6"
            tc = "white" if hot else "#333"
            ax.text(cx, cy - 0.18, str(num), ha="center", va="center",
                    fontsize=11, fontweight="bold", color=tc, zorder=4,
                    bbox=dict(boxstyle="circle,pad=0.25", fc=fc,
                              ec="#cc2200" if hot else "#aaa", linewidth=1.2))

    # ── Draw ports ───────────────────────────────────────────────────────────
    PORT_FULL_NAMES = {1: "3:1\nPort", 2: "2:1\nWood", 3: "2:1\nBrick",
                       4: "2:1\nSheep", 5: "2:1\nWheat", 6: "2:1\nOre"}
    for port in ports.values():
        px, py, pz = port["x"], port["y"], port["z"]
        ptype = port.get("type", 1)
        color = PORT_COLORS.get(ptype, "#aaa")
        label = PORT_FULL_NAMES.get(ptype, "?")

        try:
            (ax1, ay1), (ax2, ay2) = edge_endpoints(px, py, pz)
        except Exception:
            continue

        # Midpoint of port edge
        mx, my = (ax1 + ax2) / 2, (ay1 + ay2) / 2

        # Push label outward from board center
        dist = math.hypot(mx, my)
        if dist > 0:
            nx, ny = mx / dist, my / dist
        else:
            nx, ny = 0, 1
        lx, ly = mx + nx * 0.6, my + ny * 0.6

        # Lines from label to each vertex
        ax.plot([lx, ax1], [ly, ay1], color=color, linewidth=2.5,
                solid_capstyle="round", zorder=3)
        ax.plot([lx, ax2], [ly, ay2], color=color, linewidth=2.5,
                solid_capstyle="round", zorder=3)

        # Port label box
        ax.text(lx, ly, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6,
                multialignment="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=color,
                          ec="white", linewidth=1.2, alpha=0.95))

    # ── Draw vertex labels (Colonist vertex IDs) ─────────────────────────────
    for col_vid, (vx, vy) in _COL_VERTEX_POS.items():
        ax.plot(vx, vy, "o", markersize=3, color="#333",
                markeredgecolor="white", markeredgewidth=0.5, zorder=8)
        ax.text(vx, vy, str(col_vid), ha="center", va="center",
                fontsize=5.5, color="#111", zorder=9,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          edgecolor="#999", linewidth=0.5, alpha=0.85))

    plt.tight_layout()

    if save:
        out = Path(__file__).parent / "replays" / f"{game_id}_board.png"
        plt.savefig(str(out), dpi=200, bbox_inches="tight")
        print(f"Saved -> {out}")
    else:
        out = Path(__file__).parent / "replays" / f"{game_id}_board.png"
        plt.savefig(str(out), dpi=200, bbox_inches="tight")
        print(f"Saved -> {out}")
        # Try to open
        import subprocess, platform
        if platform.system() == "Windows":
            subprocess.Popen(["start", "", str(out)], shell=True)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(out)])
        else:
            subprocess.Popen(["xdg-open", str(out)])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python colonist/visualize_board.py <game_id> [save]")
        sys.exit(1)
    draw_board(sys.argv[1], save="save" in sys.argv)
