"""
env/game_state.py

Full 1v1 Catan game state: enums, data structures, constants, new_game(),
and all pure query/helper functions.

Ruleset: Colonist.io ranked 1v1
  - 15 VP to win
  - Balanced dice  (env/balanced_dice.py)
  - Friendly robber: can only target opponent with >= 3 visible VP
  - No player-to-player trades
  - Dev cards on, ports on
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from env.board import (
    Board, generate_board,
    _EDGE_LIST, _HEX_TO_VERTICES, _VERTEX_TO_HEXES,
    _EDGE_TO_HEXES, HEX_ADJACENCY,
)
from env.balanced_dice import BalancedDiceEngine


# ---------------------------------------------------------------------------
# Precomputed vertex adjacency (two vertices are adjacent iff an edge connects
# them; built once from _EDGE_LIST at import time)
# ---------------------------------------------------------------------------

_VERTEX_ADJACENCY: List[List[int]] = [[] for _ in range(54)]
for _eid, (_va, _vb) in enumerate(_EDGE_LIST):
    _VERTEX_ADJACENCY[_va].append(_vb)
    _VERTEX_ADJACENCY[_vb].append(_va)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Resource(IntEnum):
    WOOD  = 0
    BRICK = 1
    SHEEP = 2
    WHEAT = 3
    ORE   = 4


# Maps the board.py resource strings → Resource enum
_TILE_TO_RESOURCE: Dict[str, Resource] = {
    "wood":  Resource.WOOD,
    "brick": Resource.BRICK,
    "sheep": Resource.SHEEP,
    "wheat": Resource.WHEAT,
    "ore":   Resource.ORE,
}


class DevCard(IntEnum):
    KNIGHT          = 0
    ROAD_BUILDING   = 1
    YEAR_OF_PLENTY  = 2
    MONOPOLY        = 3
    VICTORY_POINT   = 4


class GamePhase(IntEnum):
    # Setup — two rounds, second round is reverse player order
    SETUP_P0_SETTLE_1 = 0
    SETUP_P0_ROAD_1   = 1
    SETUP_P1_SETTLE_1 = 2
    SETUP_P1_ROAD_1   = 3
    SETUP_P1_SETTLE_2 = 4   # reverse: P1 goes first in round 2
    SETUP_P1_ROAD_2   = 5
    SETUP_P0_SETTLE_2 = 6
    SETUP_P0_ROAD_2   = 7   # after this → ROLL
    # Main game
    ROLL              = 8
    DISCARD           = 9   # rolled 7, someone has > 7 cards
    ROBBER            = 10  # move robber (7 rolled or knight played)
    ACTIONS           = 11  # build / trade / dev card
    GAME_OVER         = 12


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROAD_COST:       Dict[Resource, int] = {Resource.WOOD: 1, Resource.BRICK: 1}
SETTLEMENT_COST: Dict[Resource, int] = {Resource.WOOD: 1, Resource.BRICK: 1,
                                         Resource.SHEEP: 1, Resource.WHEAT: 1}
CITY_COST:       Dict[Resource, int] = {Resource.WHEAT: 2, Resource.ORE: 3}
DEV_CARD_COST:   Dict[Resource, int] = {Resource.SHEEP: 1, Resource.WHEAT: 1,
                                         Resource.ORE: 1}

MAX_SETTLEMENTS = 5
MAX_CITIES      = 4
MAX_ROADS       = 15

BANK_START: Dict[Resource, int] = {r: 19 for r in Resource}

# 25 dev cards total
DEV_DECK_COMPOSITION: List[DevCard] = (
    [DevCard.KNIGHT]         * 14 +
    [DevCard.ROAD_BUILDING]  * 2  +
    [DevCard.YEAR_OF_PLENTY] * 2  +
    [DevCard.MONOPOLY]       * 2  +
    [DevCard.VICTORY_POINT]  * 5
)

LONGEST_ROAD_MIN       = 5
LARGEST_ARMY_MIN       = 3
FRIENDLY_ROBBER_MIN_VP = 3   # must have this many visible VP to be targeted
WIN_VP                 = 15


# ---------------------------------------------------------------------------
# Player state
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    player_id: int
    resources: Dict[Resource, int] = field(
        default_factory=lambda: {r: 0 for r in Resource}
    )
    # Full dev card hand (hidden from opponent)
    dev_hand: List[DevCard] = field(default_factory=list)
    # Cards bought this turn — cannot be played until next turn
    dev_bought_this_turn: List[DevCard] = field(default_factory=list)

    knights_played: int = 0

    settlements: List[int] = field(default_factory=list)   # vertex_ids
    cities:      List[int] = field(default_factory=list)   # vertex_ids
    roads:       List[int] = field(default_factory=list)   # edge_ids

    has_longest_road: bool = False
    has_largest_army: bool = False

    def resource_total(self) -> int:
        return sum(self.resources.values())

    def playable_dev_cards(self) -> List[DevCard]:
        """Cards that can be played this turn (exclude bought-this-turn)."""
        # Remove one copy of each card bought this turn from the playable set
        remaining_bought = list(self.dev_bought_this_turn)
        playable = []
        for card in self.dev_hand:
            if card in remaining_bought:
                remaining_bought.remove(card)
            else:
                playable.append(card)
        return playable


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    board:          Board
    players:        List[PlayerState]   # always length 2; index == player_id

    phase:          GamePhase
    current_player: int                 # 0 or 1
    turn_number:    int                 # increments when P0 starts their turn

    bank:           Dict[Resource, int]
    dev_deck:       List[DevCard]       # remaining; pop(-1) to draw top

    robber_hex:     int                 # current robber position

    longest_road_holder: Optional[int]
    largest_army_holder: Optional[int]

    dev_card_played_this_turn: bool
    players_to_discard:        List[int]   # who still must discard

    dice:      BalancedDiceEngine
    last_roll: Optional[int]

    winner:    Optional[int]            # None until GAME_OVER

    # vertex of the most recently placed settlement (needed for setup roads)
    last_settlement_vertex: Optional[int]

    rng: random.Random                  # non-dice randomness (shuffles, etc.)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

def new_game(seed: Optional[int] = None) -> GameState:
    """
    Create a fresh game.  Both board layout and dev deck are randomised
    from `seed` (or fully random if seed is None).
    """
    rng = random.Random(seed)

    board = generate_board(seed=rng.randint(0, 2**31 - 1))

    dev_deck = list(DEV_DECK_COMPOSITION)
    rng.shuffle(dev_deck)

    dice = BalancedDiceEngine(num_players=2, seed=rng.randint(0, 2**31 - 1))

    players = [PlayerState(player_id=i) for i in range(2)]

    return GameState(
        board=board,
        players=players,
        phase=GamePhase.SETUP_P0_SETTLE_1,
        current_player=0,
        turn_number=0,
        bank=dict(BANK_START),
        dev_deck=dev_deck,
        robber_hex=board.robber_hex,
        longest_road_holder=None,
        largest_army_holder=None,
        dev_card_played_this_turn=False,
        players_to_discard=[],
        dice=dice,
        last_roll=None,
        winner=None,
        last_settlement_vertex=None,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# VP helpers
# ---------------------------------------------------------------------------

def visible_vp(state: GameState, pid: int) -> int:
    """VP visible to all players (structures + longest road + largest army)."""
    p = state.players[pid]
    vp = len(p.settlements) + 2 * len(p.cities)
    if p.has_longest_road:
        vp += 2
    if p.has_largest_army:
        vp += 2
    return vp


def total_vp(state: GameState, pid: int) -> int:
    """True total VP including hidden VP dev cards."""
    vp_cards = sum(1 for c in state.players[pid].dev_hand
                   if c == DevCard.VICTORY_POINT)
    return visible_vp(state, pid) + vp_cards


def check_winner(state: GameState) -> Optional[int]:
    """Return player_id of winner if >= WIN_VP total VP, else None."""
    for pid in range(2):
        if total_vp(state, pid) >= WIN_VP:
            return pid
    return None


# ---------------------------------------------------------------------------
# Resource / port helpers
# ---------------------------------------------------------------------------

def can_afford(state: GameState, pid: int,
               cost: Dict[Resource, int]) -> bool:
    """True if player has at least the resources specified in cost."""
    return all(state.players[pid].resources[r] >= n
               for r, n in cost.items())


def has_port(state: GameState, pid: int, port_type: str) -> bool:
    """
    True if player has a settlement or city on a vertex with the given port.
    port_type: "3:1" or resource name ("wood", "brick", …)
    """
    p = state.players[pid]
    for vid in p.settlements + p.cities:
        if state.board.vertices[vid].port == port_type:
            return True
    return False


def trade_rate(state: GameState, pid: int, resource: Resource) -> int:
    """
    Best available trade ratio for selling this resource type.
    Returns 2 (2:1 port), 3 (3:1 port), or 4 (no port).
    """
    res_name = resource.name.lower()
    if has_port(state, pid, res_name):
        return 2
    if has_port(state, pid, "3:1"):
        return 3
    return 4


# ---------------------------------------------------------------------------
# Resource production
# ---------------------------------------------------------------------------

def production_for_roll(
    state: GameState, total: int
) -> Dict[int, Dict[Resource, int]]:
    """
    Returns {player_id: {resource: count}} of resources produced on this roll.

    Applies the standard scarcity rule: if the bank cannot cover all
    entitlements for a given resource, NO player receives that resource.
    """
    # Tally raw entitlements per resource per player
    entitlements: Dict[int, Dict[Resource, int]] = {0: {}, 1: {}}

    for h in state.board.hexes:
        if h.number_token != total or h.hex_id == state.robber_hex:
            continue
        res = _TILE_TO_RESOURCE.get(h.resource)
        if res is None:   # desert
            continue
        for vid in state.board.hex_to_vertices[h.hex_id]:
            for pid in range(2):
                p = state.players[pid]
                if vid in p.settlements:
                    entitlements[pid][res] = entitlements[pid].get(res, 0) + 1
                elif vid in p.cities:
                    entitlements[pid][res] = entitlements[pid].get(res, 0) + 2

    # Apply bank scarcity rule
    production: Dict[int, Dict[Resource, int]] = {0: {}, 1: {}}
    for res in Resource:
        total_needed = sum(entitlements[pid].get(res, 0) for pid in range(2))
        if total_needed == 0:
            continue
        if state.bank[res] >= total_needed:
            for pid in range(2):
                amt = entitlements[pid].get(res, 0)
                if amt:
                    production[pid][res] = amt
        # else: bank can't cover everyone → no one gets it (standard rule)

    return production


# ---------------------------------------------------------------------------
# Legal placement queries
# ---------------------------------------------------------------------------

def _all_occupied_vertices(state: GameState) -> set:
    """All vertex ids that have any settlement or city."""
    occupied = set()
    for p in state.players:
        occupied.update(p.settlements)
        occupied.update(p.cities)
    return occupied


def _all_road_edges(state: GameState) -> set:
    """All edge ids that have any road."""
    roads = set()
    for p in state.players:
        roads.update(p.roads)
    return roads


def legal_initial_settlement_locations(state: GameState) -> List[int]:
    """
    Valid vertices for a setup-phase settlement placement:
    - Vertex must be unoccupied
    - Distance rule: no adjacent vertex may be occupied
    """
    occupied = _all_occupied_vertices(state)
    result = []
    for vid in range(54):
        if vid in occupied:
            continue
        if any(adj in occupied for adj in _VERTEX_ADJACENCY[vid]):
            continue
        result.append(vid)
    return result


def legal_settlement_locations(state: GameState, pid: int) -> List[int]:
    """
    Valid vertices for a main-game settlement:
    - Unoccupied, distance rule satisfied
    - Adjacent to at least one of the player's own roads
    - Player has settlements remaining
    """
    p = state.players[pid]
    if len(p.settlements) >= MAX_SETTLEMENTS:
        return []

    occupied = _all_occupied_vertices(state)
    own_roads = set(p.roads)

    # Build vertex -> own road adjacency
    vertex_has_own_road: set = set()
    for eid in own_roads:
        va, vb = _EDGE_LIST[eid]
        vertex_has_own_road.add(va)
        vertex_has_own_road.add(vb)

    result = []
    for vid in range(54):
        if vid in occupied:
            continue
        if any(adj in occupied for adj in _VERTEX_ADJACENCY[vid]):
            continue
        if vid not in vertex_has_own_road:
            continue
        result.append(vid)
    return result


def legal_city_locations(state: GameState, pid: int) -> List[int]:
    """Vertices where player has a settlement (can upgrade to city)."""
    p = state.players[pid]
    if len(p.cities) >= MAX_CITIES:
        return []
    return list(p.settlements)


def legal_road_locations(
    state: GameState, pid: int,
    setup_vertex: Optional[int] = None
) -> List[int]:
    """
    Valid edges for a road placement.

    setup_vertex: during setup, roads must be adjacent to this vertex
                  (the just-placed settlement).

    Main-game rule: an empty edge is valid if at least one endpoint either:
      (a) has the player's settlement/city, OR
      (b) connects to the player's road network AND is not blocked by an
          opponent's settlement/city.
    """
    p          = state.players[pid]
    opponent   = 1 - pid
    opp        = state.players[opponent]

    if len(p.roads) >= MAX_ROADS:
        return []

    all_roads      = _all_road_edges(state)
    opp_structures = set(opp.settlements + opp.cities)
    own_structures = set(p.settlements + p.cities)

    # vertex -> set of this player's adjacent road edge ids
    vertex_own_roads: Dict[int, set] = defaultdict(set)
    for eid in p.roads:
        va, vb = _EDGE_LIST[eid]
        vertex_own_roads[va].add(eid)
        vertex_own_roads[vb].add(eid)

    result = []
    for eid, (va, vb) in enumerate(_EDGE_LIST):
        if eid in all_roads:
            continue

        if setup_vertex is not None:
            # Setup: must be adjacent to the just-placed settlement
            if va == setup_vertex or vb == setup_vertex:
                result.append(eid)
            continue

        # Main game
        for v in (va, vb):
            if v in own_structures:
                result.append(eid)
                break
            if vertex_own_roads[v] and v not in opp_structures:
                result.append(eid)
                break

    return result


def legal_robber_hexes(state: GameState, pid: int) -> List[int]:
    """
    Hexes where the robber can be placed:
    - Must differ from current robber position
    - Friendly robber: may only place on a hex with the opponent's structure
      if the opponent has >= FRIENDLY_ROBBER_MIN_VP visible VP.
    """
    opponent = 1 - pid
    opp_vp   = visible_vp(state, opponent)
    can_target_opponent = (opp_vp >= FRIENDLY_ROBBER_MIN_VP)

    opp_hexes: set = set()
    opp = state.players[opponent]
    for vid in opp.settlements + opp.cities:
        opp_hexes.update(_VERTEX_TO_HEXES[vid])

    result = []
    for hid in range(19):
        if hid == state.robber_hex:
            continue
        # If this hex has opponent structures, check friendly robber rule
        if hid in opp_hexes and not can_target_opponent:
            continue
        result.append(hid)
    return result


# ---------------------------------------------------------------------------
# Longest road
# ---------------------------------------------------------------------------

def compute_road_length(state: GameState, pid: int) -> int:
    """
    Longest continuous road for player pid.
    Uses DFS; opponent settlements/cities break road continuity.
    """
    p        = state.players[pid]
    opponent = 1 - pid
    opp      = state.players[opponent]

    if not p.roads:
        return 0

    opp_structures = frozenset(opp.settlements + opp.cities)

    # vertex -> list of this player's road edge ids
    vertex_roads: Dict[int, List[int]] = defaultdict(list)
    for eid in p.roads:
        va, vb = _EDGE_LIST[eid]
        vertex_roads[va].append(eid)
        vertex_roads[vb].append(eid)

    best = 0

    def dfs(v: int, visited: set) -> int:
        local_best = len(visited)
        for eid in vertex_roads[v]:
            if eid in visited:
                continue
            va, vb = _EDGE_LIST[eid]
            nv = vb if va == v else va
            # Can only continue through nv if opponent hasn't blocked it
            if nv in opp_structures:
                continue
            visited.add(eid)
            local_best = max(local_best, dfs(nv, visited))
            visited.remove(eid)
        return local_best

    for start_v in list(vertex_roads.keys()):
        length = dfs(start_v, set())
        if length > best:
            best = length

    return best


# ---------------------------------------------------------------------------
# Special card updates
# ---------------------------------------------------------------------------

def update_special_cards(state: GameState) -> None:
    """
    Recompute and assign Longest Road and Largest Army.
    Updates state.players[*].has_longest_road / has_largest_army in place.
    """
    # --- Longest Road ---
    lengths = [compute_road_length(state, pid) for pid in range(2)]
    holder  = state.longest_road_holder

    if holder is None:
        # Award if someone first reaches the minimum
        for pid in range(2):
            if lengths[pid] >= LONGEST_ROAD_MIN:
                holder = pid
                break
    else:
        # Challenger must strictly exceed current holder to take the card
        challenger = 1 - holder
        if lengths[challenger] > lengths[holder]:
            holder = challenger

    state.longest_road_holder = holder
    for pid in range(2):
        state.players[pid].has_longest_road = (holder == pid)

    # --- Largest Army ---
    armies = [state.players[pid].knights_played for pid in range(2)]
    army_holder = state.largest_army_holder

    if army_holder is None:
        for pid in range(2):
            if armies[pid] >= LARGEST_ARMY_MIN:
                army_holder = pid
                break
    else:
        challenger = 1 - army_holder
        if armies[challenger] > armies[army_holder]:
            army_holder = challenger

    state.largest_army_holder = army_holder
    for pid in range(2):
        state.players[pid].has_largest_army = (army_holder == pid)
