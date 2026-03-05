"""
Catan board representation, topology, and generation.

Board geometry: pointy-top hexagons in axial coordinates (q, r).
Standard Catan board: hexagonal region of radius 2 → 19 tiles,
54 intersection vertices, 72 edges.
"""

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional

__all__ = [
    "HexTile", "Vertex", "Edge", "Board",
    "generate_board",
    "HEX_ADJACENCY", "CATAN_HEX_AXIAL",
    "PORT_TYPES", "PORT_SLOT_DEFINITIONS",
]

SQRT3 = math.sqrt(3)
EPS = 1e-6  # vertex deduplication tolerance

# ---------------------------------------------------------------------------
# Axial coordinates (q, r) for the 19 Catan hex positions.
# Hexagonal region: all (q,r) with |q|<=2, |r|<=2, |q+r|<=2.
# Listed row by row (r = -2 … +2).
# ---------------------------------------------------------------------------
CATAN_HEX_AXIAL: List = [
    # row r=-2 (top, 3 hexes)
    (0, -2), (1, -2), (2, -2),
    # row r=-1 (4 hexes)
    (-1, -1), (0, -1), (1, -1), (2, -1),
    # row r=0  (middle, 5 hexes)
    (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
    # row r=+1 (4 hexes)
    (-2, 1), (-1, 1), (0, 1), (1, 1),
    # row r=+2 (bottom, 3 hexes)
    (-2, 2), (-1, 2), (0, 2),
]

# Six axial neighbor directions
AXIAL_DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1)]

# Standard tile distribution (19 total)
RESOURCE_TILES: List[str] = (
    ["wood"] * 4 + ["brick"] * 3 + ["sheep"] * 4 +
    ["wheat"] * 4 + ["ore"] * 3 + ["desert"] * 1
)

# Standard number token distribution (18 tokens for 18 non-desert hexes)
NUMBER_TOKENS: List[int] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]

# 9 port types: 5 resource-specific 2:1 ports + 4 generic 3:1 ports.
# These are shuffled randomly among the 9 fixed slot positions each game.
PORT_TYPES: List[str] = ["wood", "brick", "sheep", "wheat", "ore", "3:1", "3:1", "3:1", "3:1"]

# 9 fixed port slot positions: coastal edge IDs, clockwise from upper-left.
# Each edge ID directly identifies the two vertices that share the port.
PORT_SLOT_DEFINITIONS: List[int] = [3, 9, 29, 48, 59, 66, 63, 51, 18]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HexTile:
    hex_id: int
    resource: str       # "wood"|"brick"|"sheep"|"wheat"|"ore"|"desert"
    number_token: int   # 2-12; 0 for desert
    q: int              # axial coordinate
    r: int


@dataclass
class Vertex:
    vertex_id: int
    adjacent_hexes: List[int]   # 1-3 hex ids
    adjacent_edges: List[int]   # 2-3 edge ids
    port: Optional[str]         # "3:1"|resource name|None
    x: float                    # pixel position (for rendering)
    y: float


@dataclass
class Edge:
    edge_id: int
    vertex_a: int
    vertex_b: int
    adjacent_hexes: List[int]   # 1-2 hex ids


@dataclass
class Board:
    hexes: List[HexTile]          # len 19
    vertices: List[Vertex]        # len 54
    edges: List[Edge]             # len 72
    robber_hex: int
    hex_to_vertices: List[List[int]]  # hex_id -> ordered list of 6 vertex ids
    hex_to_edges: List[List[int]]     # hex_id -> ordered list of 6 edge ids


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _axial_to_pixel(q: int, r: int, size: float = 1.0):
    """Pointy-top hex: axial coordinates → pixel center."""
    x = size * (SQRT3 * q + SQRT3 / 2.0 * r)
    y = size * (1.5 * r)
    return x, y


def _hex_vertex_positions(q: int, r: int, size: float = 1.0) -> List:
    """Return the 6 pixel positions of a pointy-top hex at axial (q, r).
    Vertices are ordered at angles 30°, 90°, 150°, 210°, 270°, 330°."""
    cx, cy = _axial_to_pixel(q, r, size)
    return [
        (cx + size * math.cos(math.radians(30 + 60 * i)),
         cy + size * math.sin(math.radians(30 + 60 * i)))
        for i in range(6)
    ]


def _find_or_add_vertex(px: float, py: float, positions: list) -> int:
    """Return the index of an existing vertex within EPS, or append a new one."""
    for i, (vx, vy) in enumerate(positions):
        if abs(vx - px) < EPS and abs(vy - py) < EPS:
            return i
    positions.append((px, py))
    return len(positions) - 1


# ---------------------------------------------------------------------------
# Topology (computed once at module load)
# ---------------------------------------------------------------------------

def _build_topology():
    """
    Derive vertex and edge structures from the 19 axial hex positions.

    Returns:
        vertex_positions  : list of (x, y) for each vertex id
        hex_to_vertices   : list[list[int]] – 6 ordered vertex ids per hex
        vertex_to_hexes   : dict vertex_id -> list of hex ids
        edge_list         : list of (va, vb) per edge id
        edge_to_hexes     : dict edge_id -> list of hex ids
        hex_to_edges      : list[list[int]] – 6 ordered edge ids per hex
    """
    vertex_positions: list = []
    hex_to_vertices: List[List[int]] = []
    vertex_to_hexes: dict = {}
    edge_key_to_id: dict = {}
    edge_list: list = []
    edge_to_hexes: dict = {}
    hex_to_edges: List[List[int]] = []

    for hex_id, (q, r) in enumerate(CATAN_HEX_AXIAL):
        raw_verts = _hex_vertex_positions(q, r)
        v_ids = [_find_or_add_vertex(px, py, vertex_positions) for px, py in raw_verts]
        hex_to_vertices.append(v_ids)

        for vid in v_ids:
            vertex_to_hexes.setdefault(vid, [])
            if hex_id not in vertex_to_hexes[vid]:
                vertex_to_hexes[vid].append(hex_id)

        e_ids = []
        for i in range(6):
            va, vb = v_ids[i], v_ids[(i + 1) % 6]
            key = frozenset({va, vb})
            if key not in edge_key_to_id:
                eid = len(edge_list)
                edge_key_to_id[key] = eid
                edge_list.append((va, vb))
                edge_to_hexes[eid] = []
            eid = edge_key_to_id[key]
            if hex_id not in edge_to_hexes[eid]:
                edge_to_hexes[eid].append(hex_id)
            e_ids.append(eid)
        hex_to_edges.append(e_ids)

    return (
        vertex_positions,
        hex_to_vertices,
        vertex_to_hexes,
        edge_list,
        edge_to_hexes,
        hex_to_edges,
    )


(
    _VERTEX_POSITIONS,
    _HEX_TO_VERTICES,
    _VERTEX_TO_HEXES,
    _EDGE_LIST,
    _EDGE_TO_HEXES,
    _HEX_TO_EDGES,
) = _build_topology()

_COORD_TO_HEX_ID = {coords: i for i, coords in enumerate(CATAN_HEX_AXIAL)}


def _build_hex_adjacency() -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(19)]
    for i, (q, r) in enumerate(CATAN_HEX_AXIAL):
        for dq, dr in AXIAL_DIRS:
            neighbor = (q + dq, r + dr)
            if neighbor in _COORD_TO_HEX_ID:
                adj[i].append(_COORD_TO_HEX_ID[neighbor])
    return adj


HEX_ADJACENCY: List[List[int]] = _build_hex_adjacency()


# ---------------------------------------------------------------------------
# Constraint checking
# ---------------------------------------------------------------------------

def _check_constraints(hexes: List[HexTile]) -> bool:
    """Return True iff the board satisfies all Colonist.io placement rules:
      - 6 & 8 may not be adjacent
      - 2 & 12 may not be adjacent
      - Same number tokens may not be adjacent
    """
    for i, ha in enumerate(hexes):
        for j in HEX_ADJACENCY[i]:
            if j <= i:
                continue
            hb = hexes[j]
            ta, tb = ha.number_token, hb.number_token
            if ta == 0 or tb == 0:
                continue  # desert has no token
            if {ta, tb} == {6, 8}:
                return False
            if {ta, tb} == {2, 12}:
                return False
            if ta == tb:
                return False
    return True


# ---------------------------------------------------------------------------
# Board generation
# ---------------------------------------------------------------------------

def generate_board(seed=None, max_attempts: int = 100_000) -> Board:
    """
    Generate a random Catan board that satisfies Colonist.io constraints.
    Uses rejection sampling; typical acceptance rate is ~30-50%, so
    100k attempts is more than enough.

    Args:
        seed: optional RNG seed for reproducibility
        max_attempts: hard cap on rejection attempts

    Returns:
        A fully-constructed Board instance.
    """
    rng = random.Random(seed)
    resources = RESOURCE_TILES[:]
    tokens = NUMBER_TOKENS[:]

    for _ in range(max_attempts):
        rng.shuffle(resources)
        rng.shuffle(tokens)

        token_it = iter(tokens)
        hexes = [
            HexTile(
                hex_id=i,
                resource=resources[i],
                number_token=0 if resources[i] == "desert" else next(token_it),
                q=q,
                r=r,
            )
            for i, (q, r) in enumerate(CATAN_HEX_AXIAL)
        ]

        if not _check_constraints(hexes):
            continue

        # Build vertex → adjacent edge lookup
        v_to_edges: dict = {}
        for eid, (va, vb) in enumerate(_EDGE_LIST):
            v_to_edges.setdefault(va, []).append(eid)
            v_to_edges.setdefault(vb, []).append(eid)

        # Assign shuffled port types to the 9 fixed slot positions.
        # Each slot covers 2 vertices; both vertices get the same port label.
        shuffled_types = PORT_TYPES[:]
        rng.shuffle(shuffled_types)
        port_map: dict = {}   # vertex_id -> port type string
        for eid, ptype in zip(PORT_SLOT_DEFINITIONS, shuffled_types):
            va, vb = _EDGE_LIST[eid]
            port_map[va] = ptype
            port_map[vb] = ptype

        vertices = [
            Vertex(
                vertex_id=vid,
                adjacent_hexes=_VERTEX_TO_HEXES.get(vid, []),
                adjacent_edges=v_to_edges.get(vid, []),
                port=port_map.get(vid),
                x=vx,
                y=vy,
            )
            for vid, (vx, vy) in enumerate(_VERTEX_POSITIONS)
        ]

        edges = [
            Edge(
                edge_id=eid,
                vertex_a=va,
                vertex_b=vb,
                adjacent_hexes=_EDGE_TO_HEXES[eid],
            )
            for eid, (va, vb) in enumerate(_EDGE_LIST)
        ]

        desert_id = next(h.hex_id for h in hexes if h.resource == "desert")

        return Board(
            hexes=hexes,
            vertices=vertices,
            edges=edges,
            robber_hex=desert_id,
            hex_to_vertices=_HEX_TO_VERTICES,
            hex_to_edges=_HEX_TO_EDGES,
        )

    raise RuntimeError(f"Board generation failed after {max_attempts} attempts")
