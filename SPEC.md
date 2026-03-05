# SPEC — `visualization` branch

## What this branch does

Implements the foundational board module for CatanBot: the hex grid geometry, topology pre-computation, board generation with Colonist.io constraints, port assignment, and a matplotlib visualiser for manual verification.

---

## Files added / modified

| File | Description |
|---|---|
| `env/__init__.py` | Package init |
| `env/board.py` | Board module (see below) |
| `test_board.py` | 24 unit tests — all passing |
| `visualize_board.py` | matplotlib board visualiser |

---

## `env/board.py`

### Hex grid

19 hexes in pointy-top axial coordinates `(q, r)` — the hexagonal region where `|q| <= 2`, `|r| <= 2`, `|q+r| <= 2`.

```
row r=-2:  3 hexes
row r=-1:  4 hexes
row r= 0:  5 hexes
row r=+1:  4 hexes
row r=+2:  3 hexes
```

Pixel position of a hex center: `x = sqrt3*q + sqrt3/2*r`, `y = 1.5*r`.

### Topology

Pre-computed at import time from the 19 hex positions:

| Symbol | Contents |
|---|---|
| `_VERTEX_POSITIONS` | 54 `(x, y)` positions, deduplicated with EPS=1e-6 |
| `_EDGE_LIST` | 72 `(va, vb)` pairs |
| `_HEX_TO_VERTICES` | `hex_id -> [6 vertex ids]` |
| `_HEX_TO_EDGES` | `hex_id -> [6 edge ids]` |
| `_VERTEX_TO_HEXES` | `vertex_id -> [1-3 hex ids]` |
| `_EDGE_TO_HEXES` | `edge_id -> [1-2 hex ids]` |
| `HEX_ADJACENCY` | `hex_id -> [2-6 neighbour hex ids]` |

Verified by Euler characteristic: `V - E + F = 54 - 72 + 20 = 2` (F = 19 hexes + 1 outer face).

### Board generation

`generate_board(seed=None)` uses rejection sampling:
1. Shuffle 19 resource tiles and 18 number tokens
2. Assign tokens to non-desert hexes
3. Check Colonist.io constraints — reject and retry if violated:
   - 6 and 8 may not be adjacent
   - 2 and 12 may not be adjacent
   - Same number token may not appear on adjacent hexes
4. Shuffle 9 port types and assign to the 9 fixed coastal edge positions
5. Return a `Board` dataclass

Typical acceptance rate ~30-50%; 100 boards generated in ~0.1s.

### Ports

9 ports: `["wood", "brick", "sheep", "wheat", "ore", "3:1", "3:1", "3:1", "3:1"]`

Port types are shuffled each game. The 9 positions are fixed coastal edges (each adjacent to exactly 1 hex), identified by edge ID:

```python
PORT_SLOT_DEFINITIONS = [3, 9, 29, 48, 59, 66, 63, 51, 18]
```

Each port edge marks both its endpoint vertices with the port type.

### Key constants

```python
RESOURCE_TILES = ["wood"]*4 + ["brick"]*3 + ["sheep"]*4 + ["wheat"]*4 + ["ore"]*3 + ["desert"]*1
NUMBER_TOKENS  = [2,3,3,4,4,5,5,6,6,8,8,9,9,10,10,11,11,12]
PORT_TYPES     = ["wood","brick","sheep","wheat","ore","3:1","3:1","3:1","3:1"]
```

---

## `test_board.py` — test coverage

**Topology tests** (static, no board generation):
- Vertex count = 54
- Edge count = 72
- Hex count = 19
- Euler characteristic V-E+F = 2
- Each vertex adjacent to 1-3 hexes
- Each hex has exactly 6 vertices and 6 edges
- Center hex has 6 neighbours
- All hexes have >= 2 neighbours
- Adjacency is symmetric
- No duplicate edges
- Each edge has 1 or 2 adjacent hexes

**Board generation tests** (seed=42):
- Resource distribution matches standard (4/3/4/4/3/1)
- Token distribution matches standard 18 tokens
- Desert has no number token
- Robber starts on desert

**Constraint tests** (30 seeds):
- No adjacent 6 & 8
- No adjacent 2 & 12
- No adjacent same token

**Other**:
- Seeded generation is reproducible
- 100 boards generated in < 10s
- 18 port vertices (9 ports x 2)
- Ports only on coastal vertices
- Port types shuffled across seeds

---

## `visualize_board.py`

matplotlib visualiser. Shows hexes coloured by resource, number tokens with pip dots (red for 6 & 8), robber hex with bold red border, port edges as thick coloured lines with label boxes, vertex IDs, and coastal edge IDs (for debugging port positions).

```
python visualize_board.py          # random board, opens window
python visualize_board.py 42       # seed 42
python visualize_board.py 42 save  # save to board_viz.png
```
