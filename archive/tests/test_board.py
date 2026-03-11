"""
Unit tests for env/board.py.

Run: python test_board.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sys
import time
from env.board import (
    generate_board,
    HEX_ADJACENCY,
    CATAN_HEX_AXIAL,
    _VERTEX_POSITIONS,
    _EDGE_LIST,
    _VERTEX_TO_HEXES,
    _HEX_TO_VERTICES,
    _HEX_TO_EDGES,
    NUMBER_TOKENS,
    RESOURCE_TILES,
)

PASS = "  \033[92mPASS\033[0m"
FAIL = "  \033[91mFAIL\033[0m"


def run(name, fn):
    try:
        fn()
        print(f"{PASS}  {name}")
        return True
    except AssertionError as e:
        print(f"{FAIL}  {name}: {e}")
        return False
    except Exception as e:
        print(f"{FAIL}  {name}: {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Topology tests
# ---------------------------------------------------------------------------

def test_vertex_count():
    assert len(_VERTEX_POSITIONS) == 54, \
        f"Expected 54 vertices, got {len(_VERTEX_POSITIONS)}"


def test_edge_count():
    assert len(_EDGE_LIST) == 72, \
        f"Expected 72 edges, got {len(_EDGE_LIST)}"


def test_hex_count():
    assert len(CATAN_HEX_AXIAL) == 19


def test_euler_characteristic():
    # For planar graph: V - E + F = 2; F = 19 hexes + 1 outer face = 20
    V, E, F = len(_VERTEX_POSITIONS), len(_EDGE_LIST), 20
    assert V - E + F == 2, f"Euler: {V} - {E} + {F} = {V - E + F}, expected 2"


def test_vertex_hex_incidence():
    for vid, hexes in _VERTEX_TO_HEXES.items():
        assert 1 <= len(hexes) <= 3, \
            f"Vertex {vid} adjacent to {len(hexes)} hexes (expected 1-3)"


def test_each_hex_has_6_vertices_and_edges():
    for i, v_ids in enumerate(_HEX_TO_VERTICES):
        assert len(v_ids) == 6, f"Hex {i} has {len(v_ids)} vertices"
    for i, e_ids in enumerate(_HEX_TO_EDGES):
        assert len(e_ids) == 6, f"Hex {i} has {len(e_ids)} edges"


def test_center_hex_has_6_neighbors():
    center_idx = CATAN_HEX_AXIAL.index((0, 0))
    assert len(HEX_ADJACENCY[center_idx]) == 6


def test_all_hexes_have_at_least_2_neighbors():
    for i, adj in enumerate(HEX_ADJACENCY):
        assert len(adj) >= 2, f"Hex {i} has only {len(adj)} neighbors"


def test_adjacency_is_symmetric():
    for i, adj in enumerate(HEX_ADJACENCY):
        for j in adj:
            assert i in HEX_ADJACENCY[j], \
                f"Hex {i} lists {j} as neighbor but not vice-versa"


def test_edges_are_unique():
    keys = [frozenset({va, vb}) for va, vb in _EDGE_LIST]
    assert len(keys) == len(set(keys)), "Duplicate edges found"


def test_each_edge_has_1_or_2_adjacent_hexes():
    from env.board import _EDGE_TO_HEXES
    for eid, hexes in _EDGE_TO_HEXES.items():
        assert 1 <= len(hexes) <= 2, \
            f"Edge {eid} adjacent to {len(hexes)} hexes"


# ---------------------------------------------------------------------------
# Board generation tests
# ---------------------------------------------------------------------------

def test_resource_distribution():
    board = generate_board(seed=42)
    counts = {}
    for h in board.hexes:
        counts[h.resource] = counts.get(h.resource, 0) + 1
    expected = {"wood": 4, "brick": 3, "sheep": 4, "wheat": 4, "ore": 3, "desert": 1}
    assert counts == expected, f"Got: {counts}"


def test_token_distribution():
    board = generate_board(seed=42)
    tokens = sorted(h.number_token for h in board.hexes if h.number_token > 0)
    expected = sorted(NUMBER_TOKENS)
    assert tokens == expected, f"Got: {tokens}"


def test_desert_has_no_token():
    board = generate_board(seed=42)
    for h in board.hexes:
        if h.resource == "desert":
            assert h.number_token == 0


def test_robber_starts_on_desert():
    board = generate_board(seed=42)
    assert board.hexes[board.robber_hex].resource == "desert", \
        f"Robber on {board.hexes[board.robber_hex].resource}"


def test_constraint_no_6_8_adjacent():
    for seed in range(30):
        board = generate_board(seed=seed)
        for i, ha in enumerate(board.hexes):
            for j in HEX_ADJACENCY[i]:
                if j <= i:
                    continue
                hb = board.hexes[j]
                ta, tb = ha.number_token, hb.number_token
                assert {ta, tb} != {6, 8}, \
                    f"seed={seed}: 6&8 adjacent at hexes {i},{j}"


def test_constraint_no_2_12_adjacent():
    for seed in range(30):
        board = generate_board(seed=seed)
        for i, ha in enumerate(board.hexes):
            for j in HEX_ADJACENCY[i]:
                if j <= i:
                    continue
                hb = board.hexes[j]
                ta, tb = ha.number_token, hb.number_token
                assert {ta, tb} != {2, 12}, \
                    f"seed={seed}: 2&12 adjacent at hexes {i},{j}"


def test_constraint_no_same_number_adjacent():
    for seed in range(30):
        board = generate_board(seed=seed)
        for i, ha in enumerate(board.hexes):
            for j in HEX_ADJACENCY[i]:
                if j <= i:
                    continue
                hb = board.hexes[j]
                ta, tb = ha.number_token, hb.number_token
                if ta == 0 or tb == 0:
                    continue
                assert ta != tb, \
                    f"seed={seed}: token {ta} adjacent to itself at hexes {i},{j}"


def test_board_vertex_edge_counts():
    board = generate_board(seed=0)
    assert len(board.vertices) == 54
    assert len(board.edges) == 72


def test_reproducibility():
    b1 = generate_board(seed=7)
    b2 = generate_board(seed=7)
    for h1, h2 in zip(b1.hexes, b2.hexes):
        assert h1.resource == h2.resource and h1.number_token == h2.number_token


def test_port_count_and_types():
    board = generate_board(seed=42)
    port_vertices = [v for v in board.vertices if v.port is not None]
    assert len(port_vertices) == 18, \
        f"Expected 18 port vertices (9 ports x 2), got {len(port_vertices)}"
    type_counts = {}
    for v in port_vertices:
        type_counts[v.port] = type_counts.get(v.port, 0) + 1
    # Each port type should appear exactly on 2 vertices
    for ptype, count in type_counts.items():
        assert count % 2 == 0, f"Port type {ptype!r} appears on odd number of vertices"
    # Exactly 5 resource ports and 4 generic 3:1 ports
    resource_ports = sum(1 for v in port_vertices if v.port != "3:1") // 2
    generic_ports  = sum(1 for v in port_vertices if v.port == "3:1") // 2
    assert resource_ports == 5, f"Expected 5 resource ports, got {resource_ports}"
    assert generic_ports  == 4, f"Expected 4 generic 3:1 ports, got {generic_ports}"


def test_ports_on_coastal_vertices_only():
    from env.board import _EDGE_TO_HEXES, _EDGE_LIST
    board = generate_board(seed=7)
    for v in board.vertices:
        if v.port is None:
            continue
        # This vertex must touch at least one coastal edge
        coastal = any(len(_EDGE_TO_HEXES[eid]) == 1 for eid in v.adjacent_edges)
        assert coastal, f"Vertex {v.vertex_id} has port but is not coastal"


def test_port_types_shuffled_across_seeds():
    b1 = generate_board(seed=1)
    b2 = generate_board(seed=2)
    ports1 = sorted(v.port for v in b1.vertices if v.port)
    ports2 = sorted(v.port for v in b2.vertices if v.port)
    assert ports1 == ports2, "Port type totals differ across seeds"
    # Verify positions actually differ (shuffled)
    pos1 = {v.vertex_id: v.port for v in b1.vertices if v.port}
    pos2 = {v.vertex_id: v.port for v in b2.vertices if v.port}
    assert pos1 != pos2, "Port positions identical across seeds — shuffle not working"


def test_generation_speed():
    start = time.time()
    for i in range(100):
        generate_board(seed=i)
    elapsed = time.time() - start
    assert elapsed < 10.0, f"100 boards took {elapsed:.2f}s (too slow)"
    print(f"  (100 boards in {elapsed:.3f}s)", end="")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

TESTS = [
    ("54 vertices in topology",          test_vertex_count),
    ("72 edges in topology",             test_edge_count),
    ("19 hexes in layout",               test_hex_count),
    ("Euler characteristic V-E+F=2",     test_euler_characteristic),
    ("Each vertex adjacent to 1-3 hexes",test_vertex_hex_incidence),
    ("Each hex has 6 vertices & edges",  test_each_hex_has_6_vertices_and_edges),
    ("Center hex has 6 neighbors",       test_center_hex_has_6_neighbors),
    ("All hexes have >=2 neighbors",     test_all_hexes_have_at_least_2_neighbors),
    ("Hex adjacency is symmetric",       test_adjacency_is_symmetric),
    ("Edges are unique",                 test_edges_are_unique),
    ("Each edge has 1-2 adjacent hexes", test_each_edge_has_1_or_2_adjacent_hexes),
    ("Resource distribution correct",   test_resource_distribution),
    ("Number token distribution correct",test_token_distribution),
    ("Desert has no token",             test_desert_has_no_token),
    ("Robber starts on desert",          test_robber_starts_on_desert),
    ("No 6 & 8 adjacent (30 seeds)",     test_constraint_no_6_8_adjacent),
    ("No 2 & 12 adjacent (30 seeds)",    test_constraint_no_2_12_adjacent),
    ("No same token adjacent (30 seeds)",test_constraint_no_same_number_adjacent),
    ("Board has 54 vertices, 72 edges",  test_board_vertex_edge_counts),
    ("Seeded generation is reproducible",test_reproducibility),
    ("100 boards generated in <10s",     test_generation_speed),
    ("Port count: 18 vertices (9x2)",    test_port_count_and_types),
    ("Ports only on coastal vertices",   test_ports_on_coastal_vertices_only),
    ("Port types shuffled across seeds", test_port_types_shuffled_across_seeds),
]

if __name__ == "__main__":
    print("env/board.py tests\n")
    passed = sum(run(name, fn) for name, fn in TESTS)
    total = len(TESTS)
    print(f"\n{passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
