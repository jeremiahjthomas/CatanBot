
"""
edge_table.py

Generates a table of initial road placements where the Colonist edge ID is
still ambiguous, along with:
  - the settlement vertex (Colonist + CatanEnv)
  - the 2-3 candidate CatanEnv edges

Run alongside the visualize_mapping.py visualization to identify each edge.
Also saves a board image with CatanEnv edge IDs labeled.

Usage:
    python colonist/edge_table.py              # print table + save edge_ids.png
    python colonist/edge_table.py --game 214191014   # focus on one game
"""
import sys
import math
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env.board import _EDGE_LIST, _VERTEX_POSITIONS, CATAN_HEX_AXIAL, SQRT3

# ── Vertex mapping ──────────────────────────────────────────────────────────
COL_TO_CATAN_V = {
    0:48, 1:47, 2:42, 3:38, 4:39, 5:49, 6:30, 7:25, 8:26, 9:40, 10:15, 11:16,
    12:28, 13:27, 14:14, 15:1, 16:2, 17:17, 18:0, 19:35, 20:4, 21:3, 22:7,
    23:6, 24:9, 25:8, 26:11, 27:10, 28:13, 29:12, 30:23, 31:22, 32:24, 33:20,
    34:36, 35:5, 36:37, 37:33, 38:46, 39:45, 40:34, 41:43, 42:53, 43:52, 44:44,
    45:50, 46:51, 47:41, 48:32, 49:29, 50:19, 51:18, 52:21, 53:31,
}

# vertex -> adjacent CatanEnv edge IDs
vertex_to_edges: dict = {}
for _eid, (_va, _vb) in enumerate(_EDGE_LIST):
    vertex_to_edges.setdefault(_va, set()).add(_eid)
    vertex_to_edges.setdefault(_vb, set()).add(_eid)

# Known mappings (Colonist edge ID -> CatanEnv edge ID) — loaded from JSON
_mapping_file = Path(__file__).parent / "colonist_to_catan_mapping.json"
KNOWN: dict = {int(k): v for k, v in json.loads(_mapping_file.read_text()).get("col_to_catan_edge", {}).items()}

# ── Board image with edge IDs ────────────────────────────────────────────────
def save_edge_id_board(out_path="colonist/edge_ids.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("CatanEnv Edge IDs", fontsize=14, pad=10)

    SIZE = 1.0
    RESOURCE_COLORS = {
        "wood":   "#5d8a3c",
        "brick":  "#c0522a",
        "sheep":  "#90c040",
        "wheat":  "#e8c840",
        "ore":    "#909090",
        "desert": "#e0d090",
    }

    from env.board import generate_board
    board = generate_board(seed=42)

    # Draw hexes
    for hx in board.hexes:
        cx = SIZE * (SQRT3 * hx.q + SQRT3 / 2 * hx.r)
        cy = SIZE * 1.5 * hx.r
        verts = [(cx + SIZE * math.cos(math.radians(30 + 60*i)),
                  cy + SIZE * math.sin(math.radians(30 + 60*i)))
                 for i in range(6)]
        poly = Polygon(verts, closed=True,
                       facecolor=RESOURCE_COLORS.get(hx.resource, "#ccc"),
                       edgecolor="#555", linewidth=0.8, alpha=0.7)
        ax.add_patch(poly)

    # Draw edges with ID labels
    for eid, (va, vb) in enumerate(_EDGE_LIST):
        xa, ya = _VERTEX_POSITIONS[va]
        xb, yb = _VERTEX_POSITIONS[vb]
        mx, my = (xa+xb)/2, (ya+yb)/2
        ax.plot([xa, xb], [ya, yb], color="#333", linewidth=1.5, zorder=2)
        ax.text(mx, my, str(eid), ha="center", va="center",
                fontsize=5.5, fontweight="bold",
                color="white", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="#222", ec="none", alpha=0.75))

    # Draw vertices
    for vid, (vx, vy) in enumerate(_VERTEX_POSITIONS):
        ax.plot(vx, vy, "o", color="#888", markersize=3, zorder=3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"Saved edge ID board -> {out_path}")


# ── Extract initial road placements ─────────────────────────────────────────
def get_initial_roads(events):
    """
    Returns list of (col_edge_id, col_vertex, round_num) for each player's
    first two roads (initial placement).
    col_vertex = the settlement placed BEFORE this road.
    """
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

            road_num = len(player_edges[owner])  # 0 = first road, 1 = second
            if road_num < 2:
                # The settlement this road must touch
                corner_idx = road_num  # 0 = first settlement, 1 = second
                corners = player_corners.get(owner, [])
                if corner_idx < len(corners):
                    col_v = corners[corner_idx]
                    result.append((col_e, col_v, road_num + 1, owner))

            player_edges[owner].append(col_e)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", type=str, default=None,
                        help="Focus on a single game ID")
    parser.add_argument("--no-image", action="store_true",
                        help="Skip saving the board image")
    args = parser.parse_args()

    if not args.no_image:
        save_edge_id_board()

    replays_dir = Path(__file__).parent / "replays"
    if args.game:
        files = [replays_dir / f"replay_{args.game}_gamedata.json"]
    else:
        files = sorted(replays_dir.glob("replay_*_gamedata.json"))

    # Collect all initial road entries across games
    rows = []
    for f in files:
        if not f.exists():
            print(f"Not found: {f}")
            continue
        gid = f.stem.split("_")[1]
        events = json.loads(f.read_text())["data"]["eventHistory"]["events"]
        for col_e, col_v, rnd, owner in get_initial_roads(events):
            catan_v = COL_TO_CATAN_V.get(col_v)
            if catan_v is None:
                continue
            adj = sorted(vertex_to_edges.get(catan_v, set()))
            known_mapping = KNOWN.get(col_e)
            rows.append({
                "game": gid,
                "round": rnd,
                "player": owner,
                "col_edge": col_e,
                "col_vertex": col_v,
                "catan_vertex": catan_v,
                "adj_catan_edges": adj,
                "known": known_mapping,
            })

    # Group by col_edge - show each Colonist edge once with all its observations
    from collections import defaultdict
    by_edge = defaultdict(list)
    for r in rows:
        by_edge[r["col_edge"]].append(r)

    # Print table: only unresolved (not in KNOWN) or with >1 candidate
    print(f"\n{'Ce':>4}  {'Known':>6}  {'Candidates':30}  Context (game / round / settlement)")
    print("-" * 90)

    for col_e in sorted(by_edge.keys()):
        entries = by_edge[col_e]
        known = KNOWN.get(col_e)
        # Intersect all candidate sets across observations
        candidate_sets = [set(e["adj_catan_edges"]) for e in entries]
        candidates = candidate_sets[0]
        for s in candidate_sets[1:]:
            candidates &= s
        if not candidates:
            candidates = candidate_sets[0]  # fallback if over-constrained

        # Remove other known mappings from candidates
        solved_catan = set(KNOWN.values())
        filtered = candidates - (solved_catan - ({known} if known else set()))
        if not filtered:
            filtered = candidates

        status = f"E{known}" if known is not None else "?"
        cand_str = str(sorted(filtered))

        context = "; ".join(
            f"g{e['game']} r{e['round']} C{e['col_vertex']}=E{e['catan_vertex']}"
            for e in entries[:2]
        )
        print(f"Ce{col_e:>3}  {status:>6}  {cand_str:30}  {context}")


if __name__ == "__main__":
    main()
