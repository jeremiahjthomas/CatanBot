"""
Derive Colonist edge ID -> CatanEnv edge ID using hex-coordinate geometry.

1. For each CatanEnv edge (va,vb), compute the Colonist hex edge coordinate (hx,hy,z)
   via: CatanEnv vertex -> Colonist vertex -> hex corner -> hex edge
2. Build a hex_edge_coord -> CatanEnv_edge_ID lookup (bijective)
3. For each road event in replays, determine which hex edges are adjacent to the
   player's settlement corners. Constrain Colonist edge ID -> set of hex edge coords.
4. Propagate to uniqueness.
"""
import json
import sys
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env.board import _EDGE_LIST

SQRT3 = math.sqrt(3)

# Complete vertex mapping (Colonist -> CatanEnv)
COL_TO_CATAN = {
    0:48, 1:47, 2:42, 3:38, 4:39, 5:49, 6:30, 7:25, 8:26, 9:40, 10:15, 11:16,
    12:28, 13:27, 14:14, 15:1, 16:2, 17:17, 18:0, 19:35, 20:4, 21:3, 22:7,
    23:6, 24:9, 25:8, 26:11, 27:10, 28:13, 29:12, 30:23, 31:22, 32:24, 33:20,
    34:36, 35:5, 36:37, 37:33, 38:46, 39:45, 40:34, 41:43, 42:53, 43:52, 44:44,
    45:50, 46:51, 47:41, 48:32, 49:29, 50:19, 51:18, 52:21, 53:31,
}
CATAN_TO_COL = {v: k for k, v in COL_TO_CATAN.items()}

# Colonist vertex -> hex corner (hx, hy, z) where z=1:North, z=0:South
# Derived from vertex positions: North of (hx,hy) at (sqrt3*hx+sqrt3/2*hy, 1.5*hy+1)
#                                 South of (hx,hy) at same x, cy-1
COL_CORNER = {
    0:(-2,2,1), 1:(-2,3,0), 2:(-1,1,1), 3:(-2,2,0), 4:(-2,1,1), 5:(-3,3,0),
    6:(-1,0,1), 7:(-2,1,0), 8:(-2,0,1), 9:(-3,2,0), 10:(-1,-1,1), 11:(-2,0,0),
    12:(-2,-1,1), 13:(-3,1,0), 14:(-1,0,0), 15:(0,-2,1), 16:(-1,-1,0), 17:(-1,-2,1),
    18:(0,-1,0), 19:(2,1,0), 20:(0,-2,0), 21:(0,-3,1), 22:(1,-2,1), 23:(1,-1,0),
    24:(2,-3,1), 25:(1,-2,0), 26:(2,-2,1), 27:(2,-1,0), 28:(3,-3,1), 29:(2,-2,0),
    30:(2,-1,1), 31:(2,0,0), 32:(3,-2,1), 33:(1,0,0), 34:(2,0,1), 35:(1,-3,1),
    36:(3,-1,1), 37:(1,1,0), 38:(1,1,1), 39:(1,2,0), 40:(1,0,1), 41:(0,2,0),
    42:(0,2,1), 43:(0,3,0), 44:(0,1,1), 45:(-1,3,0), 46:(-1,2,1), 47:(-1,2,0),
    48:(0,0,1), 49:(-1,1,0), 50:(0,-1,1), 51:(0,0,0), 52:(1,-1,1), 53:(0,1,0),
}

def corners_to_edge(ca, cb):
    """Return hex edge (hx,hy,z) from two hex corners, or None if not adjacent."""
    hxa,hya,za = ca
    hxb,hyb,zb = cb
    # NW (z=2): North(hx,hy) <-> South(hx-1,hy+1)
    if za==1 and zb==0 and hxb==hxa-1 and hyb==hya+1: return (hxa,hya,2)
    if za==0 and zb==1 and hxa==hxb-1 and hya==hyb+1: return (hxb,hyb,2)
    # W (z=1): South(hx-1,hy+1) <-> North(hx,hy-1)  [FIXED: return (hx,hy,1) not (hx,hy+2,1)]
    if za==0 and zb==1 and hxb==hxa+1 and hyb==hya-2: return (hxa+1,hya-1,1)
    if za==1 and zb==0 and hxa==hxb+1 and hya==hyb-2: return (hxb+1,hyb-1,1)
    # SW (z=0): North(hx,hy-1) <-> South(hx,hy)
    if za==1 and zb==0 and hxa==hxb and hya==hyb-1: return (hxb,hyb,0)
    if za==0 and zb==1 and hxb==hxa and hyb==hya-1: return (hxa,hya,0)
    return None

def col_vertex_adjacent_hex_edges(col_v):
    """All hex edge coordinates adjacent to Colonist vertex col_v."""
    hx, hy, z = COL_CORNER[col_v]
    if z == 1:  # North vertex
        # Adjacent edges: NW(hx,hy,2), SW(hx,hy+1,0), W(hx,hy+1,1)
        return {(hx,hy,2), (hx,hy+1,0), (hx,hy+1,1)}
    else:  # South vertex (z=0)
        # Adjacent edges: NW(hx+1,hy-1,2), SW(hx,hy,0), W(hx+1,hy-1,1)
        return {(hx+1,hy-1,2), (hx,hy,0), (hx+1,hy-1,1)}

# Build: hex_edge_coord -> CatanEnv edge ID (bijective)
hex_to_catan_edge = {}
catan_edge_to_hex = {}
conflicts = []

for eid, (va, vb) in enumerate(_EDGE_LIST):
    col_va = CATAN_TO_COL.get(va)
    col_vb = CATAN_TO_COL.get(vb)
    if col_va is None or col_vb is None:
        print(f"MISSING vertex mapping for CatanEnv edge {eid}: va={va} col={col_va}, vb={vb} col={col_vb}")
        continue
    ca = COL_CORNER[col_va]
    cb = COL_CORNER[col_vb]
    hex_e = corners_to_edge(ca, cb)
    if hex_e is None:
        print(f"No hex edge for CatanEnv edge {eid}: col_va={col_va}{ca} col_vb={col_vb}{cb}")
        continue
    if hex_e in hex_to_catan_edge:
        conflicts.append((hex_e, hex_to_catan_edge[hex_e], eid))
    else:
        hex_to_catan_edge[hex_e] = eid
        catan_edge_to_hex[eid] = hex_e

if conflicts:
    print(f"CONFLICTS in hex->CatanEnv: {conflicts}")

print(f"Mapped {len(hex_to_catan_edge)} hex edges to CatanEnv edges (should be 72)")

# Filter: only hex edges that are actually on the board
board_hex_edges = set(hex_to_catan_edge.keys())
print(f"Board hex edge coordinates: {len(board_hex_edges)}")

# Known: Colonist edge ID -> hex edge coord
KNOWN_COL_TO_HEX = {
    21:(0,-1,0), 26:(1,-1,0), 35:(2,-2,2), 40:(2,-1,2),
    7:(-1,0,2),  71:(0,1,0),  47:(2,1,0),  58:(-1,2,0),
}
# Verify known against computed
for col_eid, hex_e in KNOWN_COL_TO_HEX.items():
    catan_e = hex_to_catan_edge.get(hex_e, "MISSING")
    known_catan = {21:0,26:6,35:12,40:28,7:37,71:39,47:45,58:53}[col_eid]
    status = "OK" if catan_e == known_catan else f"MISMATCH (expected E{known_catan})"
    print(f"  Ce{col_eid:2d} -> hex{hex_e} -> E{catan_e} [{status}]")

print()

# ---------------------------------------------------------------
# Constraint propagation in hex-edge-coord space
# col_edge_id -> set of possible hex edge coordinates
# ---------------------------------------------------------------
col_hex_candidates = {col_eid: {hex_e} for col_eid, hex_e in KNOWN_COL_TO_HEX.items()}

def apply_hex_constraint(col_eid, allowed_hex: set, source=""):
    if col_eid not in col_hex_candidates:
        col_hex_candidates[col_eid] = set(board_hex_edges)
    new = col_hex_candidates[col_eid] & allowed_hex
    if new:
        col_hex_candidates[col_eid] = new
    else:
        pass  # constraint violated, keep old

# Process replays
all_files = sorted((Path(__file__).parent / "replays").glob("replay_*_gamedata.json"))
for f in all_files:
    data = json.loads(f.read_text())
    events = data["data"]["eventHistory"]["events"]
    gid = f.stem.split("_")[1]

    player_corners = {}    # pid -> list of col_vertex
    player_edges   = {}    # pid -> list of col_edge
    last_corner    = {}    # pid -> most recently placed col_vertex
    prev_ev_had_corner = {}  # pid -> bool: did prev event add a corner for this player?

    for i, ev in enumerate(events):
        ms = ev.get("stateChange", {}).get("mapState", {})

        # Track which players placed a corner this event
        corners_added = {}
        for k, v in ms.get("tileCornerStates", {}).items():
            col_v = int(k)
            owner = v.get("owner")
            if owner is not None:
                player_corners.setdefault(owner, [])
                if col_v not in player_corners[owner]:
                    player_corners[owner].append(col_v)
                    last_corner[owner] = col_v
                    corners_added[owner] = col_v

        for k, v in ms.get("tileEdgeStates", {}).items():
            col_e = int(k)
            owner = v.get("owner")
            if owner is None:
                continue
            player_edges.setdefault(owner, [])
            if col_e in player_edges[owner]:
                continue

            accessible_hex_edges = set()

            # TIGHT constraint: if the previous event placed a corner for this owner,
            # use only that corner (initial placement rule: road must touch new settlement)
            recent_corner = last_corner.get(owner)
            if recent_corner is not None and recent_corner in player_corners.get(owner, []):
                # Check if the most recent corner was placed in the previous event or this event
                just_placed = corners_added.get(owner)
                if just_placed == recent_corner:
                    # Road in same event as settlement - tight constraint
                    tight_verts = {recent_corner}
                elif len(player_corners.get(owner, [])) <= 2 and len(player_edges.get(owner, [])) <= 1:
                    # Early game: road is likely adjacent to the most recent settlement
                    tight_verts = {recent_corner}
                else:
                    tight_verts = None
            else:
                tight_verts = None

            if tight_verts:
                # Use only the most recent corner for tight constraint
                for col_v in tight_verts:
                    if col_v in COL_CORNER:
                        adj = col_vertex_adjacent_hex_edges(col_v)
                        accessible_hex_edges.update(adj & board_hex_edges)
            else:
                # Loose constraint: union of all accessible vertices
                for col_v in player_corners.get(owner, []):
                    if col_v in COL_CORNER:
                        adj = col_vertex_adjacent_hex_edges(col_v)
                        accessible_hex_edges.update(adj & board_hex_edges)
                # From solved roads
                for prev_e in player_edges.get(owner, []):
                    if prev_e in col_hex_candidates and len(col_hex_candidates[prev_e]) == 1:
                        prev_hex_e = list(col_hex_candidates[prev_e])[0]
                        prev_catan_e = hex_to_catan_edge.get(prev_hex_e)
                        if prev_catan_e is not None:
                            va, vb = _EDGE_LIST[prev_catan_e]
                            for col_v in [CATAN_TO_COL.get(va), CATAN_TO_COL.get(vb)]:
                                if col_v is not None and col_v in COL_CORNER:
                                    adj = col_vertex_adjacent_hex_edges(col_v)
                                    accessible_hex_edges.update(adj & board_hex_edges)

            if accessible_hex_edges:
                apply_hex_constraint(col_e, accessible_hex_edges, f"g{gid}_ev{i}_tight={'tight' if tight_verts else 'loose'}")

            player_edges[owner].append(col_e)

# Propagation
def propagate():
    for _ in range(500):
        changed = False
        solved = {ce: list(cands)[0] for ce, cands in col_hex_candidates.items() if len(cands) == 1}
        solved_hexes = set(solved.values())
        for ce, cands in col_hex_candidates.items():
            if len(cands) > 1:
                new = cands - solved_hexes
                if new and len(new) < len(cands):
                    col_hex_candidates[ce] = new
                    changed = True
        if not changed:
            break

propagate()

# Convert to CatanEnv edge IDs
col_to_catan_edge = {}
for col_eid, hex_cands in col_hex_candidates.items():
    catan_cands = set()
    for hx in hex_cands:
        ce = hex_to_catan_edge.get(hx)
        if ce is not None:
            catan_cands.add(ce)
    col_to_catan_edge[col_eid] = catan_cands

solved = {ce: list(cands)[0] for ce, cands in col_to_catan_edge.items() if len(cands) == 1}
unsolved = {ce: sorted(cands) for ce, cands in col_to_catan_edge.items() if len(cands) != 1}
unseen = sorted(set(range(72)) - set(col_to_catan_edge.keys()))

print(f"Solved: {len(solved)}/72  |  Unsolved: {len(unsolved)}  |  Unseen: {len(unseen)}")
print(f"Unseen: {unseen}")

# Check bijection
from collections import Counter
ctr = Counter(solved.values())
dupes = {k:v for k,v in ctr.items() if v>1}
if dupes:
    print(f"Bijection violations: {dupes}")
    for catan_e, cnt in sorted(dupes.items()):
        claimants = [ce for ce, v in solved.items() if v == catan_e]
        print(f"  E{catan_e} claimed by Ce{claimants}")
else:
    print("Bijection OK for solved edges.")

print()
print("=== Ambiguous (<=5 candidates) ===")
for ce in sorted(unsolved):
    cands = unsolved[ce]
    if len(cands) <= 5:
        # Also show hex coords
        hex_cands = sorted(col_hex_candidates[ce])
        print(f"  Ce{ce:2d} -> E{cands} | hex{hex_cands}")

print()
print("=== All solved ===")
for ce in sorted(solved):
    src = "known" if ce in KNOWN_COL_TO_HEX else "derived"
    print(f"  Ce{ce:2d} -> E{solved[ce]:2d}  {src}")

assigned = set(solved.values())
unassigned = sorted(set(range(72)) - assigned)
print(f"\nUnassigned CatanEnv edges: {unassigned}")
