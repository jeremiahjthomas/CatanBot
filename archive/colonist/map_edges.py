"""
map_edges.py

Interactive CLI to manually map Colonist edge IDs -> CatanEnv edge IDs.

For each unresolved Colonist edge:
  - Opens a highlight image showing the board with:
      * Settlement vertex highlighted in yellow
      * Candidate CatanEnv edges colored and labeled
  - Prompts you to type which CatanEnv edge it is
  - Saves the answer to colonist_to_catan_mapping.json

Usage:
    python colonist/map_edges.py
"""
import sys
import math
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env.board import _EDGE_LIST, _VERTEX_POSITIONS, CATAN_HEX_AXIAL, SQRT3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# ── Constants ────────────────────────────────────────────────────────────────
MAPPING_FILE = Path(__file__).parent / "colonist_to_catan_mapping.json"
HIGHLIGHT_DIR = Path(__file__).parent / "highlights"
HIGHLIGHT_DIR.mkdir(exist_ok=True)

SIZE = 1.0
RESOURCE_COLORS = {
    "wood":   "#5d8a3c",
    "brick":  "#c0522a",
    "sheep":  "#90c040",
    "wheat":  "#e8c840",
    "ore":    "#909090",
    "desert": "#e0d090",
}
CANDIDATE_COLORS = ["#e63946", "#2196f3", "#4caf50", "#ff9800", "#9c27b0"]

COL_TO_CATAN_V = {
    0:48, 1:47, 2:42, 3:38, 4:39, 5:49, 6:30, 7:25, 8:26, 9:40, 10:15, 11:16,
    12:28, 13:27, 14:14, 15:1, 16:2, 17:17, 18:0, 19:35, 20:4, 21:3, 22:7,
    23:6, 24:9, 25:8, 26:11, 27:10, 28:13, 29:12, 30:23, 31:22, 32:24, 33:20,
    34:36, 35:5, 36:37, 37:33, 38:46, 39:45, 40:34, 41:43, 42:53, 43:52, 44:44,
    45:50, 46:51, 47:41, 48:32, 49:29, 50:19, 51:18, 52:21, 53:31,
}

vertex_to_edges: dict = {}
for _eid, (_va, _vb) in enumerate(_EDGE_LIST):
    vertex_to_edges.setdefault(_va, set()).add(_eid)
    vertex_to_edges.setdefault(_vb, set()).add(_eid)

# ── Load/save mapping ────────────────────────────────────────────────────────
def load_mapping():
    data = json.loads(MAPPING_FILE.read_text())
    col_to_catan_edge = {int(k): v for k, v in data.get("col_to_catan_edge", {}).items()}
    return data, col_to_catan_edge

def save_mapping(data, col_to_catan_edge):
    data["col_to_catan_edge"] = {str(k): v for k, v in sorted(col_to_catan_edge.items())}
    MAPPING_FILE.write_text(json.dumps(data, indent=2))

# ── Build unresolved edge list from replays ──────────────────────────────────
def get_initial_roads(events):
    player_corners = {}
    player_edges = {}
    result = []
    for ev in events:
        ms = ev.get("stateChange", {}).get("mapState", {})
        for k, v in ms.get("tileCornerStates", {}).items():
            col_v = int(k)
            owner = v.get("owner")
            if owner:
                player_corners.setdefault(owner, [])
                if col_v not in player_corners[owner]:
                    player_corners[owner].append(col_v)
        for k, v in ms.get("tileEdgeStates", {}).items():
            col_e = int(k)
            owner = v.get("owner")
            if owner is None:
                continue
            player_edges.setdefault(owner, [])
            if col_e in player_edges[owner]:
                continue
            road_num = len(player_edges[owner])
            if road_num < 2:
                corners = player_corners.get(owner, [])
                if road_num < len(corners):
                    result.append((col_e, corners[road_num], road_num + 1, owner))
            player_edges[owner].append(col_e)
    return result

def collect_candidates(col_to_catan_edge):
    """Returns list of (col_e, catan_v, candidates, context_str) for unresolved edges."""
    replays_dir = Path(__file__).parent / "replays"
    solved_catan = set(col_to_catan_edge.values())

    # Accumulate candidate sets per colonist edge across all games
    by_edge = {}  # col_e -> {catan_v, candidates (intersection), context}
    for f in sorted(replays_dir.glob("replay_*_gamedata.json")):
        gid = f.stem.split("_")[1]
        events = json.loads(f.read_text())["data"]["eventHistory"]["events"]
        for col_e, col_v, rnd, owner in get_initial_roads(events):
            if col_e in col_to_catan_edge:
                continue
            catan_v = COL_TO_CATAN_V.get(col_v)
            if catan_v is None:
                continue
            adj = set(vertex_to_edges.get(catan_v, set()))
            if col_e not in by_edge:
                by_edge[col_e] = {"catan_v": catan_v, "candidates": adj,
                                  "context": f"g{gid} r{rnd} C{col_v}=E{catan_v}"}
            else:
                by_edge[col_e]["candidates"] &= adj

    # Filter out already-solved catan edges from candidates
    result = []
    for col_e, info in sorted(by_edge.items()):
        candidates = sorted(info["candidates"] - (solved_catan - {col_to_catan_edge.get(col_e)}))
        if not candidates:
            candidates = sorted(info["candidates"])
        result.append((col_e, info["catan_v"], candidates, info["context"]))
    return result

# ── Highlight image ──────────────────────────────────────────────────────────
def make_highlight(col_e, catan_v, candidates, out_path):
    from env.board import generate_board
    board = generate_board(seed=42)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"Ce{col_e}  —  settlement vertex E{catan_v}  —  candidates: {candidates}",
                 fontsize=11, pad=10)

    # Draw hexes
    for hx in board.hexes:
        cx = SIZE * (SQRT3 * hx.q + SQRT3 / 2 * hx.r)
        cy = SIZE * 1.5 * hx.r
        verts = [(cx + SIZE * math.cos(math.radians(30 + 60*i)),
                  cy + SIZE * math.sin(math.radians(30 + 60*i)))
                 for i in range(6)]
        poly = Polygon(verts, closed=True,
                       facecolor=RESOURCE_COLORS.get(hx.resource, "#ccc"),
                       edgecolor="#555", linewidth=0.8, alpha=0.6)
        ax.add_patch(poly)

    # Draw all edges grey
    for eid, (va, vb) in enumerate(_EDGE_LIST):
        xa, ya = _VERTEX_POSITIONS[va]
        xb, yb = _VERTEX_POSITIONS[vb]
        ax.plot([xa, xb], [ya, yb], color="#aaa", linewidth=1.2, zorder=2)

    # Draw candidate edges with colors + labels
    for i, eid in enumerate(candidates):
        color = CANDIDATE_COLORS[i % len(CANDIDATE_COLORS)]
        va, vb = _EDGE_LIST[eid]
        xa, ya = _VERTEX_POSITIONS[va]
        xb, yb = _VERTEX_POSITIONS[vb]
        ax.plot([xa, xb], [ya, yb], color=color, linewidth=5, zorder=3, alpha=0.85)
        mx, my = (xa+xb)/2, (ya+yb)/2
        ax.text(mx, my, f"E{eid}", ha="center", va="center",
                fontsize=10, fontweight="bold", color="white", zorder=5,
                bbox=dict(boxstyle="round,pad=0.2", fc=color, ec="none"))

    # Draw settlement vertex in yellow
    vx, vy = _VERTEX_POSITIONS[catan_v]
    ax.plot(vx, vy, "o", color="#FFD600", markersize=14, zorder=6,
            markeredgecolor="#333", markeredgewidth=1.5)
    ax.text(vx, vy + 0.18, f"E{catan_v}", ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#333", zorder=7)

    # Draw all other vertices small
    for vid, (vx2, vy2) in enumerate(_VERTEX_POSITIONS):
        if vid != catan_v:
            ax.plot(vx2, vy2, "o", color="#888", markersize=3, zorder=4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

# ── Main interactive loop ────────────────────────────────────────────────────
def main():
    import subprocess, platform

    data, col_to_catan_edge = load_mapping()
    todos = collect_candidates(col_to_catan_edge)

    if not todos:
        print("All edges resolved!")
        return

    print(f"\n{len(todos)} unresolved Colonist edges. Type a number to assign, 's' to skip, 'q' to quit.\n")

    for idx, (col_e, catan_v, candidates, context) in enumerate(todos):
        img_path = HIGHLIGHT_DIR / f"Ce{col_e}.png"
        make_highlight(col_e, catan_v, candidates, img_path)

        # Open image
        if platform.system() == "Windows":
            subprocess.Popen(["start", "", str(img_path)], shell=True)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(img_path)])
        else:
            subprocess.Popen(["xdg-open", str(img_path)])

        print(f"[{idx+1}/{len(todos)}] Ce{col_e}")
        print(f"  Settlement : C{COL_TO_CATAN_V.get(catan_v,'?')} -> CatanEnv vertex {catan_v}")
        print(f"  Candidates : {candidates}  ({context})")
        print(f"  Image      : {img_path}")

        while True:
            ans = input("  Which CatanEnv edge? > ").strip().lower()
            if ans == "q":
                save_mapping(data, col_to_catan_edge)
                print("Saved. Exiting.")
                return
            if ans == "s":
                print("  Skipped.")
                break
            try:
                val = int(ans)
                if val not in candidates:
                    confirm = input(f"  {val} is not in {candidates}. Use anyway? [y/N] ").strip().lower()
                    if confirm != "y":
                        continue
                col_to_catan_edge[col_e] = val
                save_mapping(data, col_to_catan_edge)
                print(f"  Saved: Ce{col_e} -> E{val}")
                break
            except ValueError:
                print("  Enter a number, 's' to skip, or 'q' to quit.")

    print(f"\nDone! {len(col_to_catan_edge)} edges mapped total.")
    print(f"Saved to {MAPPING_FILE}")

if __name__ == "__main__":
    main()
