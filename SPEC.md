# SPEC — `gameState` branch

## What this branch does

Implements the full 1v1 Catan game state: all enums, data structures,
constants, the `new_game()` constructor, and every pure query/helper
function needed to drive the game. No action application logic yet —
that lives in the next branch (`catanEnv`).

---

## Files added

| File | Description |
|---|---|
| `env/game_state.py` | Game state module (see below) |
| `test_game_state.py` | 57 unit tests — all passing |

---

## `env/game_state.py`

### Enums

| Enum | Values |
|---|---|
| `Resource` | WOOD, BRICK, SHEEP, WHEAT, ORE |
| `DevCard` | KNIGHT, ROAD_BUILDING, YEAR_OF_PLENTY, MONOPOLY, VICTORY_POINT |
| `GamePhase` | 8 setup sub-phases + ROLL, DISCARD, ROBBER, ACTIONS, GAME_OVER |

Setup phases follow the standard Catan 2-round placement order:
P0 settle → P0 road → P1 settle → P1 road → P1 settle (reverse) →
P1 road → P0 settle → P0 road → ROLL.

### Constants

| Constant | Value |
|---|---|
| `ROAD_COST` | wood×1, brick×1 |
| `SETTLEMENT_COST` | wood×1, brick×1, sheep×1, wheat×1 |
| `CITY_COST` | wheat×2, ore×3 |
| `DEV_CARD_COST` | sheep×1, wheat×1, ore×1 |
| `MAX_SETTLEMENTS / CITIES / ROADS` | 5 / 4 / 15 |
| `BANK_START` | 19 of each resource |
| `DEV_DECK_COMPOSITION` | 14 knights + 2 road building + 2 YoP + 2 monopoly + 5 VP = 25 |
| `LONGEST_ROAD_MIN` | 5 |
| `LARGEST_ARMY_MIN` | 3 |
| `FRIENDLY_ROBBER_MIN_VP` | 3 |
| `WIN_VP` | 15 |

### Data structures

```python
@dataclass
class PlayerState:
    player_id: int
    resources: Dict[Resource, int]       # hand
    dev_hand: List[DevCard]              # full hand (hidden from opponent)
    dev_bought_this_turn: List[DevCard]  # cannot be played until next turn
    knights_played: int
    settlements: List[int]               # vertex_ids
    cities: List[int]                    # vertex_ids
    roads: List[int]                     # edge_ids
    has_longest_road: bool
    has_largest_army: bool

@dataclass
class GameState:
    board: Board
    players: List[PlayerState]           # len 2
    phase: GamePhase
    current_player: int
    turn_number: int
    bank: Dict[Resource, int]
    dev_deck: List[DevCard]
    robber_hex: int
    longest_road_holder: Optional[int]
    largest_army_holder: Optional[int]
    dev_card_played_this_turn: bool
    players_to_discard: List[int]
    dice: BalancedDiceEngine
    last_roll: Optional[int]
    winner: Optional[int]
    last_settlement_vertex: Optional[int]
    rng: random.Random
```

### Constructor

```python
state = new_game(seed=42)
```

Generates a fresh board, shuffles the dev deck, and initialises a
`BalancedDiceEngine` — all from `seed`.

### Query functions

| Function | Returns |
|---|---|
| `visible_vp(state, pid)` | VP visible to all players |
| `total_vp(state, pid)` | True VP including hidden VP cards |
| `check_winner(state)` | player_id or None |
| `can_afford(state, pid, cost)` | bool |
| `has_port(state, pid, port_type)` | bool |
| `trade_rate(state, pid, resource)` | 2, 3, or 4 |
| `production_for_roll(state, total)` | `{pid: {resource: count}}` |
| `legal_initial_settlement_locations(state)` | vertex_ids (setup) |
| `legal_settlement_locations(state, pid)` | vertex_ids (main game) |
| `legal_city_locations(state, pid)` | vertex_ids |
| `legal_road_locations(state, pid, setup_vertex=None)` | edge_ids |
| `legal_robber_hexes(state, pid)` | hex_ids |
| `compute_road_length(state, pid)` | int (DFS longest path) |
| `update_special_cards(state)` | mutates state in place |

### Key design decisions

**Vertex adjacency** (`_VERTEX_ADJACENCY`) is precomputed at import time
from `_EDGE_LIST` — used by all placement legality checks.

**Distance rule**: a settlement vertex is only legal if none of its
directly adjacent vertices are occupied.

**Road length** uses DFS with backtracking over the player's road graph.
Opponent settlements/cities block traversal through shared vertices.
Tie-breaking: current holder keeps the card on a tie; challenger must
strictly exceed to take it.

**Scarcity rule**: if the bank cannot cover all entitlements for a
resource on a given roll, no player receives that resource.

**Friendly robber**: `legal_robber_hexes` filters out hexes containing
opponent structures unless the opponent has `>= 3` visible VP.

**Dev card timing**: `PlayerState.playable_dev_cards()` excludes cards
in `dev_bought_this_turn` — correctly handles the "can't play what you
just bought" rule.

---

## `test_game_state.py` — test coverage (57 tests)

- `new_game()`: phase, player, board, bank, dev deck, robber, winner,
  reproducibility
- VP: visible vs total, longest road, largest army, VP dev cards,
  win detection
- `can_afford`: exact resources, insufficient resources
- `has_port` / `trade_rate`: 2:1, 3:1, no-port
- `legal_initial_settlement_locations`: all 54 at start, distance rule
- `legal_settlement_locations`: requires road adjacency, distance rule,
  occupied vertex excluded
- `legal_city_locations`: only own settlements, city limit
- `legal_road_locations`: setup mode, main game, opponent-blocked vertex
- `legal_robber_hexes`: current hex excluded, friendly robber filter
- `production_for_roll`: settlement/city production, robber block, empty bank
- `compute_road_length`: 0 roads, N-road chain, broken by opponent
- `update_special_cards`: army/road thresholds, tie retention, transfer
