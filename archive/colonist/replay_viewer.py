"""
replay_viewer.py

Interactive step-through of road/settlement placements from a Colonist replay.

Controls:
  ← / → arrow keys   — previous / next step
  Left / Right buttons — same

Usage:
    python colonist/replay_viewer.py 214191014
"""

import sys
import math
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.widgets import Button

from env.board import _VERTEX_POSITIONS, _EDGE_LIST

# ── Mappings ──────────────────────────────────────────────────────────────────
_MAPPING_FILE = Path(__file__).parent / "colonist_to_catan_mapping.json"
_mapping = json.loads(_MAPPING_FILE.read_text())

_COL_TO_CATAN_V = {int(k): v for k, v in _mapping["col_to_catan_vertex"].items()}
_CATAN_TO_COL_V = {v: int(k) for k, v in _mapping["col_to_catan_vertex"].items()}
_COL_TO_CATAN_E = {int(k): v for k, v in _mapping.get("col_to_catan_edge", {}).items()}

# Colonist-space positions (negate y because colonist y-down, CatanEnv y-up)
_COL_VERTEX_POS: dict[int, tuple] = {}
for catan_vid, (vx, vy) in enumerate(_VERTEX_POSITIONS):
    col_vid = _CATAN_TO_COL_V.get(catan_vid)
    if col_vid is not None:
        _COL_VERTEX_POS[col_vid] = (vx, -vy)

def col_edge_pos(col_eid: int):
    """Midpoint of a Colonist edge in colonist render space."""
    catan_eid = _COL_TO_CATAN_E.get(col_eid)
    if catan_eid is None:
        return None
    va, vb = _EDGE_LIST[catan_eid]
    ax_, ay_ = _VERTEX_POSITIONS[va]
    bx_, by_ = _VERTEX_POSITIONS[vb]
    return ((ax_ + bx_) / 2, -(ay_ + by_) / 2)

def col_edge_endpoints(col_eid: int):
    """Both endpoint positions for a Colonist edge in colonist render space."""
    catan_eid = _COL_TO_CATAN_E.get(col_eid)
    if catan_eid is None:
        return None
    va, vb = _EDGE_LIST[catan_eid]
    ax_, ay_ = _VERTEX_POSITIONS[va]
    bx_, by_ = _VERTEX_POSITIONS[vb]
    return (ax_, -ay_), (bx_, -by_)

# ── Board constants ───────────────────────────────────────────────────────────
RESOURCE_NAMES  = {0: "desert", 1: "wood", 2: "brick", 3: "sheep", 4: "wheat", 5: "ore"}
RESOURCE_COLORS = {
    "desert": "#e0d090", "wood":  "#5d8a3c", "brick": "#c0522a",
    "sheep":  "#90c040", "wheat": "#e8c840", "ore":   "#909090",
}
PORT_COLORS = {1: "#dddddd", 2: "#5d8a3c", 3: "#c0522a", 4: "#90c040", 5: "#e8c840", 6: "#909090"}
PORT_FULL_NAMES = {1: "3:1\nPort", 2: "2:1\nWood", 3: "2:1\nBrick",
                   4: "2:1\nSheep", 5: "2:1\nWheat", 6: "2:1\nOre"}
HOT_NUMBERS = {6, 8}
SIZE   = 1.0
SQRT3  = math.sqrt(3)

PLAYER_COLORS = {1: "#e74c3c", 5: "#2575c4"}   # red / blue


def hex_center(x, y):
    return SIZE * (SQRT3 * x + SQRT3 / 2 * y), SIZE * 1.5 * (-y)

def hex_corners(x, y):
    cx, cy = hex_center(x, y)
    return [(cx + SIZE * math.cos(math.radians(30 + 60 * i)),
             cy + SIZE * math.sin(math.radians(30 + 60 * i))) for i in range(6)]

def edge_endpoints_tile(x, y, z):
    corners = hex_corners(x, y)
    z_to_corners = {0: (1, 2), 1: (2, 3), 2: (3, 4)}
    i, j = z_to_corners[z % 3]
    return corners[i], corners[j]


# ── Parse replay into steps ───────────────────────────────────────────────────
def parse_steps(game_id: str):
    f = Path(__file__).parent / "replays" / f"replay_{game_id}_gamedata.json"
    if not f.exists():
        print(f"File not found: {f}")
        sys.exit(1)
    data   = json.loads(f.read_text())
    replay = data["data"]["eventHistory"]
    map_state = replay["initialState"]["mapState"]
    events = replay["events"]

    steps = []
    for ev in events:
        ms      = ev.get("stateChange", {}).get("mapState", {})
        corners = {int(k): v for k, v in ms.get("tileCornerStates", {}).items()}
        edges   = {int(k): v for k, v in ms.get("tileEdgeStates",   {}).items()}
        if not corners and not edges:
            continue

        desc_parts = []
        for vid, info in corners.items():
            btype = "city" if info.get("buildingType") == 2 else "settlement"
            pname = f"P{info['owner']}"
            desc_parts.append(f"{pname} {btype} v{vid}")
        for eid, info in edges.items():
            pname = f"P{info['owner']}"
            desc_parts.append(f"{pname} road e{eid}")

        steps.append({
            "corners": corners,
            "edges":   edges,
            "desc":    "  +  ".join(desc_parts),
        })

    return map_state, steps


# ── Drawing ───────────────────────────────────────────────────────────────────
def draw_board_background(ax, map_state):
    """Draw static board: hexes, numbers, ports."""
    tiles = map_state["tileHexStates"]
    ports = map_state.get("portEdgeStates", {})

    for tile in tiles.values():
        x, y   = tile["x"], tile["y"]
        rtype  = RESOURCE_NAMES.get(tile["type"], "desert")
        color  = RESOURCE_COLORS[rtype]
        corners = hex_corners(x, y)
        ax.add_patch(Polygon(corners, closed=True, facecolor=color,
                             edgecolor="#555", linewidth=1.2, zorder=1))
        cx, cy = hex_center(x, y)
        ax.text(cx, cy + 0.22, rtype.upper(), ha="center", va="center",
                fontsize=6.5, color="white" if rtype in ("wood", "ore", "brick") else "#333",
                fontweight="bold", zorder=3)
        num = tile["diceNumber"]
        if num > 0:
            hot = num in HOT_NUMBERS
            fc  = "#cc2200" if hot else "#fffbe6"
            tc  = "white"   if hot else "#333"
            ax.text(cx, cy - 0.18, str(num), ha="center", va="center",
                    fontsize=11, fontweight="bold", color=tc, zorder=4,
                    bbox=dict(boxstyle="circle,pad=0.25", fc=fc,
                              ec="#cc2200" if hot else "#aaa", linewidth=1.2))

    for port in ports.values():
        px, py, pz = port["x"], port["y"], port["z"]
        ptype = port.get("type", 1)
        color = PORT_COLORS.get(ptype, "#aaa")
        label = PORT_FULL_NAMES.get(ptype, "?")
        try:
            (ax1, ay1), (ax2, ay2) = edge_endpoints_tile(px, py, pz)
        except Exception:
            continue
        mx, my = (ax1 + ax2) / 2, (ay1 + ay2) / 2
        dist = math.hypot(mx, my)
        nx, ny = (mx / dist, my / dist) if dist > 0 else (0, 1)
        lx, ly = mx + nx * 0.6, my + ny * 0.6
        ax.plot([lx, ax1], [ly, ay1], color=color, linewidth=2.5,
                solid_capstyle="round", zorder=3)
        ax.plot([lx, ax2], [ly, ay2], color=color, linewidth=2.5,
                solid_capstyle="round", zorder=3)
        ax.text(lx, ly, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=6,
                multialignment="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=color,
                          ec="white", linewidth=1.2, alpha=0.95))

    # Vertex ID labels
    for col_vid, (vx, vy) in _COL_VERTEX_POS.items():
        ax.plot(vx, vy, "o", markersize=2.5, color="#555",
                markeredgecolor="white", markeredgewidth=0.4, zorder=7)
        ax.text(vx, vy, str(col_vid), ha="center", va="center",
                fontsize=4.5, color="#444", zorder=8,
                bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                          edgecolor="#bbb", linewidth=0.4, alpha=0.8))


def draw_placements(ax, cumulative_corners, cumulative_edges):
    """Draw all roads and settlements placed so far."""
    # Roads
    for col_eid, info in cumulative_edges.items():
        ep = col_edge_endpoints(col_eid)
        if ep is None:
            continue
        (ax_, ay_), (bx_, by_) = ep
        color = PLAYER_COLORS.get(info["owner"], "#888")
        ax.plot([ax_, bx_], [ay_, by_], color=color, linewidth=6,
                solid_capstyle="round", zorder=9, alpha=0.85)

    # Settlements / cities
    for col_vid, info in cumulative_corners.items():
        pos = _COL_VERTEX_POS.get(col_vid)
        if pos is None:
            continue
        vx, vy = pos
        color = PLAYER_COLORS.get(info["owner"], "#888")
        btype = info.get("buildingType", 1)
        if btype == 2:  # city
            mk, ms_ = "s", 14
        else:           # settlement
            mk, ms_ = "^", 11
        ax.plot(vx, vy, mk, markersize=ms_, color=color,
                markeredgecolor="white", markeredgewidth=1.2, zorder=10)


# ── Interactive viewer ────────────────────────────────────────────────────────
class ReplayViewer:
    def __init__(self, game_id: str):
        self.map_state, self.steps = parse_steps(game_id)
        self.game_id  = game_id
        self.n_steps  = len(self.steps)
        self.current  = 0   # 0 = starting board, 1..n_steps = after step i

        # Build cumulative state snapshots
        self.cum_corners: list[dict] = [{}]
        self.cum_edges:   list[dict] = [{}]
        for step in self.steps:
            cc = dict(self.cum_corners[-1])
            ce = dict(self.cum_edges[-1])
            cc.update(step["corners"])
            ce.update(step["edges"])
            self.cum_corners.append(cc)
            self.cum_edges.append(ce)

        self._build_figure()
        self._render()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(13, 13))
        # Main board axes
        self.ax = self.fig.add_axes([0.02, 0.08, 0.96, 0.88])
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        # Button axes
        ax_prev = self.fig.add_axes([0.3,  0.01, 0.16, 0.05])
        ax_next = self.fig.add_axes([0.54, 0.01, 0.16, 0.05])
        self.btn_prev = Button(ax_prev, "◀  Prev")
        self.btn_next = Button(ax_next, "Next  ▶")
        self.btn_prev.on_clicked(lambda _: self._go(-1))
        self.btn_next.on_clicked(lambda _: self._go(+1))

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def _on_key(self, event):
        if event.key == "right":
            self._go(+1)
        elif event.key == "left":
            self._go(-1)

    def _go(self, delta):
        self.current = max(0, min(self.n_steps, self.current + delta))
        self._render()

    def _render(self):
        self.ax.cla()
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        draw_board_background(self.ax, self.map_state)
        draw_placements(self.ax, self.cum_corners[self.current],
                        self.cum_edges[self.current])

        if self.current == 0:
            title = f"Game {self.game_id} — starting board  (step 0 / {self.n_steps})"
        else:
            step = self.steps[self.current - 1]
            title = (f"Game {self.game_id} — step {self.current} / {self.n_steps}"
                     f"\n{step['desc']}")
        self.ax.set_title(title, fontsize=11, pad=8)

        # Legend
        patches = [
            mpatches.Patch(color=PLAYER_COLORS[1], label="Player 1"),
            mpatches.Patch(color=PLAYER_COLORS[5], label="Player 5"),
        ]
        self.ax.legend(handles=patches, loc="lower right", fontsize=8,
                       framealpha=0.88, edgecolor="#ccc")

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python colonist/replay_viewer.py <game_id>")
        sys.exit(1)
    ReplayViewer(sys.argv[1]).show()
