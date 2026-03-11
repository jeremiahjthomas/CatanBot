"""
heuristic_players.py

Three named placement strategies for 1v1 Catan, usable as full game opponents.

Strategies
----------
road_builder  — prioritise wood & brick pips; expand fast with roads/settlements
ows           — prioritise ore, wheat, sheep; buy dev cards, aim for Largest Army
balanced      — maximise total pips with a strong diversity bonus

Each strategy is a complete opponent: it handles setup placement AND all
main-game phases (roll, discard, robber, actions).  The main-game logic is
the same for all three strategies; only the build priority order and the
settlement-scoring weights differ.

Public API
----------
    action_id = heuristic_action(state, strategy_name, rng)

strategy_name: "road_builder" | "ows" | "balanced"
rng:           np.random.Generator  (for tie-breaking / fallback)
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from typing import Optional, Set

import numpy as np

from env.game_state import (
    GameState, GamePhase, Resource, DevCard,
    legal_initial_settlement_locations, legal_settlement_locations,
    legal_city_locations, legal_road_locations,
    can_afford, trade_rate,
    ROAD_COST, SETTLEMENT_COST, CITY_COST, DEV_CARD_COST,
    _VERTEX_ADJACENCY,
)
from env.actions import (
    action_mask,
    SETTLE_START, CITY_START, ROAD_START, ROLL,
    ROBBER_START, BUY_DEV, PLAY_KNIGHT, PLAY_RB,
    YOP_START, MONO_START, TRADE_START, END_TURN, DISCARD_START,
    YOP_COMBOS, TRADE_COMBOS,
)
from env.board import _EDGE_LIST

# ---------------------------------------------------------------------------
# Pip value table (standard Catan probabilities × 36)
# ---------------------------------------------------------------------------

PIP = {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

# Port types that the OWS strategy values for reachability bonus
_OWS_VALUED_PORTS = {"wheat", "sheep", "3:1"}

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
#
# res_weights   : multiplier applied to each resource's pip value when scoring
#                 a vertex.  Higher = the strategy values that resource more.
# diversity_wt  : bonus per NEW resource type the vertex adds (not already
#                 accessible by existing settlements/cities).
# build_order   : priority order for main-game building actions.

STRATEGIES = {
    "road_builder": {
        # Wood & brick are everything for roads + early settlements.
        "res_weights": {
            "wood":  2.5,
            "brick": 2.5,
            "sheep": 0.5,   # need sheep for settlements but not priority
            "wheat": 0.4,
            "ore":   0.1,
        },
        "diversity_wt": 1.0,
        # Build order: settlements first, then roads for Longest Road,
        # then cities, dev cards as low priority.
        "build_order": [
            "place_settlement",
            "place_road",
            "place_city",
            "buy_dev_card",
        ],
    },

    "ows": {
        # Ore + wheat for cities; sheep for dev cards; ignore wood/brick.
        # Setup vertex scoring uses _ows_score_vertex (guide formula), not res_weights.
        # res_weights below are used only for main-game discard decisions.
        "res_weights": {
            "wood":  0.1,
            "brick": 0.1,
            "sheep": 1.8,
            "wheat": 2.2,
            "ore":   2.2,
        },
        "diversity_wt": 0.8,
        # Buy dev cards ASAP; upgrade to cities; roads only if free.
        "build_order": [
            "buy_dev_card",
            "place_city",
            "place_settlement",
            "place_road",
        ],
    },

    "balanced": {
        # Equal weights + strong diversity bonus.
        "res_weights": {
            "wood":  1.0,
            "brick": 1.0,
            "sheep": 1.0,
            "wheat": 1.0,
            "ore":   1.0,
        },
        "diversity_wt": 3.5,   # heavily reward accessing new resource types
        "build_order": [
            "place_city",
            "place_settlement",
            "buy_dev_card",
            "place_road",
        ],
    },
}


# ---------------------------------------------------------------------------
# Setup-phase vertex scoring
# ---------------------------------------------------------------------------

def _accessible_resources(state: GameState, pid: int) -> Set[str]:
    """Resource types the player can produce from existing settlements/cities."""
    p = state.players[pid]
    res: Set[str] = set()
    for vid in p.settlements + p.cities:
        for hid in state.board.vertices[vid].adjacent_hexes:
            h = state.board.hexes[hid]
            if h.resource != "desert" and PIP.get(h.number_token, 0) > 0:
                res.add(h.resource)
    return res


def _port_reachable(state: GameState, vid: int, port_types: set, max_hops: int = 2) -> bool:
    """Return True if a port of the given type(s) is reachable from vid within max_hops edges."""
    visited = {vid}
    frontier = {vid}
    for _ in range(max_hops):
        next_f = set()
        for v in frontier:
            for adj in _VERTEX_ADJACENCY[v]:
                if adj not in visited:
                    visited.add(adj)
                    next_f.add(adj)
        frontier = next_f
    return any(state.board.vertices[v].port in port_types for v in visited)


def _ows_score_vertex(
    state: GameState,
    vid: int,
    already_have: Set[str],
) -> float:
    """
    Score a vertex using the OWS-priority formula from the placement guide.

    Base score = 3.0*Pw + 2.5*Po + 2.0*Ps + 1.3*Pb + 1.3*Pwd

    Bonuses (from placement_strategy.md Section 2.2):
      +2  if vertex touches both a wheat tile AND an ore tile (OWS core)
      +1  if vertex touches two or more wheat tiles
      +1  if vertex touches two or more ore tiles
      +2  if a 6 or 8 wheat tile is present AND any ore tile is present
      +1  if a wheat, sheep, or 3:1 port is reachable within 2 roads
      +0.8 per new resource type not already in player's production
    """
    v = state.board.vertices[vid]
    pw = po = ps = pb = pwd = 0.0
    has_wheat = has_ore = False
    has_high_wheat = False   # 6 or 8 wheat tile
    wheat_tile_count = ore_tile_count = 0
    new_resources: Set[str] = set()

    for hid in v.adjacent_hexes:
        h = state.board.hexes[hid]
        if h.resource == "desert":
            continue
        pip = PIP.get(h.number_token, 0)
        if pip == 0:
            continue
        r = h.resource
        if r == "wheat":
            pw += pip
            has_wheat = True
            wheat_tile_count += 1
            if h.number_token in (6, 8):
                has_high_wheat = True
        elif r == "ore":
            po += pip
            has_ore = True
            ore_tile_count += 1
        elif r == "sheep":
            ps += pip
        elif r == "brick":
            pb += pip
        elif r == "wood":
            pwd += pip
        if r not in already_have:
            new_resources.add(r)

    # Base OWS-weighted score
    score = 3.0*pw + 2.5*po + 2.0*ps + 1.3*pb + 1.3*pwd

    # Bonuses
    if has_wheat and has_ore:
        score += 2.0
    if wheat_tile_count >= 2:
        score += 1.0
    if ore_tile_count >= 2:
        score += 1.0
    if has_high_wheat and has_ore:
        score += 2.0  # premium city-quality spot
    if _port_reachable(state, vid, _OWS_VALUED_PORTS, max_hops=2):
        score += 1.0
    score += 0.8 * len(new_resources)  # mild diversity bonus

    return score


def _score_vertex(
    state: GameState,
    vid: int,
    res_weights: dict,
    diversity_wt: float,
    already_have: Set[str],
) -> float:
    """
    Score a vertex for settlement placement.

    score = sum over adjacent hexes of (pip * res_weight)
            + diversity_wt * (number of new resource types this vertex adds)
            + small port bonus
    """
    v = state.board.vertices[vid]
    total = 0.0
    new_resources: Set[str] = set()

    for hid in v.adjacent_hexes:
        h = state.board.hexes[hid]
        if h.resource == "desert" or hid == state.robber_hex:
            continue
        pip = PIP.get(h.number_token, 0)
        if pip == 0:
            continue
        w = res_weights.get(h.resource, 1.0)
        total += pip * w
        if h.resource not in already_have:
            new_resources.add(h.resource)

    total += diversity_wt * len(new_resources)

    if v.port is not None:
        total += 1.5   # small bonus for port access

    return total


def _best_settle(state: GameState, strategy: str, legal_vids: list[int]) -> int:
    """Pick the highest-scoring legal settlement vertex."""
    pid          = state.current_player
    already_have = _accessible_resources(state, pid)

    best_score = -1.0
    best_vid   = legal_vids[0]
    for vid in legal_vids:
        if strategy == "ows":
            s = _ows_score_vertex(state, vid, already_have)
        else:
            cfg = STRATEGIES[strategy]
            s = _score_vertex(state, vid, cfg["res_weights"],
                              cfg["diversity_wt"], already_have)
        if s > best_score:
            best_score = s
            best_vid   = vid
    return best_vid


# ---------------------------------------------------------------------------
# Setup-phase road scoring
# ---------------------------------------------------------------------------

def _score_road_direction(
    state: GameState,
    far_vid: int,
    strategy: str,
    already_have: Set[str],
) -> float:
    """
    Score a setup road by the potential of the vertex at its far end.

    We look TWO hops out: the far vertex itself, plus the best vertex
    reachable from the far vertex in one additional step (to capture
    expansion potential beyond the immediate neighbour).
    """
    occupied = {v for p in state.players for v in p.settlements + p.cities}

    if strategy == "ows":
        immediate = _ows_score_vertex(state, far_vid, already_have)
        extension = 0.0
        for adj_vid in _VERTEX_ADJACENCY[far_vid]:
            if adj_vid in occupied:
                continue
            s = _ows_score_vertex(state, adj_vid, already_have)
            if s > extension:
                extension = s
    else:
        cfg = STRATEGIES[strategy]
        immediate = _score_vertex(state, far_vid, cfg["res_weights"],
                                  cfg["diversity_wt"], already_have)
        extension = 0.0
        for adj_vid in _VERTEX_ADJACENCY[far_vid]:
            if adj_vid in occupied:
                continue
            s = _score_vertex(state, adj_vid, cfg["res_weights"],
                              cfg["diversity_wt"], already_have)
            if s > extension:
                extension = s

    # Blend: immediate vertex counts more than distant extension
    return immediate + 0.4 * extension


def _best_road_setup(state: GameState, strategy: str, legal_eids: list[int]) -> int:
    """Pick setup road that points toward the best future expansion."""
    pid          = state.current_player
    already_have = _accessible_resources(state, pid)
    last_vid     = state.last_settlement_vertex   # settlement we just placed

    best_score = -1.0
    best_eid   = legal_eids[0]
    for eid in legal_eids:
        va, vb = _EDGE_LIST[eid]
        far_vid = vb if va == last_vid else va
        s = _score_road_direction(state, far_vid, strategy, already_have)
        if s > best_score:
            best_score = s
            best_eid   = eid
    return best_eid


# ---------------------------------------------------------------------------
# Main-game helpers
# ---------------------------------------------------------------------------

def _best_city(state: GameState, pid: int) -> Optional[int]:
    """Upgrade the settlement with highest pip production."""
    locs = legal_city_locations(state, pid)
    if not locs:
        return None
    best_pip = -1
    best_vid = locs[0]
    for vid in locs:
        pip = sum(
            PIP.get(state.board.hexes[hid].number_token, 0)
            for hid in state.board.vertices[vid].adjacent_hexes
            if hid != state.robber_hex
        )
        if pip > best_pip:
            best_pip = pip
            best_vid = vid
    return best_vid


def _best_main_settle(state: GameState, pid: int, strategy: str) -> Optional[int]:
    locs = legal_settlement_locations(state, pid)
    if not locs:
        return None
    cfg          = STRATEGIES[strategy]
    already_have = _accessible_resources(state, pid)
    return _best_settle(state, strategy, locs)


def _best_main_road(state: GameState, pid: int, strategy: str) -> Optional[int]:
    """
    Pick a road that extends toward the best future settlement,
    prioritising edges adjacent to existing settlements/cities.
    """
    locs = legal_road_locations(state, pid)
    if not locs:
        return None
    already_have = _accessible_resources(state, pid)
    p            = state.players[pid]
    structures   = set(p.settlements + p.cities)

    best_score = -1.0
    best_eid   = locs[0]
    for eid in locs:
        va, vb = _EDGE_LIST[eid]
        # Score from the perspective of each endpoint that is NOT a structure
        for far in (va, vb):
            if far in structures:
                continue
            s = _score_road_direction(state, far, strategy, already_have)
            if s > best_score:
                best_score = s
                best_eid   = eid
    return best_eid


def _trade_for_build(state: GameState, pid: int, strategy: str) -> Optional[int]:
    """
    Try one bank trade that gets us 1 step closer to a priority build.
    Returns action_id or None.
    """
    cfg         = STRATEGIES[strategy]
    build_order = cfg["build_order"]
    p           = state.players[pid]

    # Determine what we're trying to build next
    target_cost = None
    for build_type in build_order:
        if build_type == "place_city" and legal_city_locations(state, pid):
            target_cost = CITY_COST
            break
        if build_type == "place_settlement" and legal_settlement_locations(state, pid):
            target_cost = SETTLEMENT_COST
            break
        if build_type == "buy_dev_card" and state.dev_deck:
            target_cost = DEV_CARD_COST
            break
        if build_type == "place_road" and legal_road_locations(state, pid):
            target_cost = ROAD_COST
            break

    if target_cost is None:
        return None

    # Shortage: resources we still need
    shortage = {r: max(0, need - p.resources[r])
                for r, need in target_cost.items()}
    if sum(shortage.values()) == 0:
        return None   # can already afford it

    # Find needed resource we're closest to having via trade
    needed = max(shortage, key=shortage.get)

    # Pick surplus resource to trade away (not needed for target, have most of)
    mask   = action_mask(state)
    for i, (give_r, recv_r) in enumerate(TRADE_COMBOS):
        if recv_r != needed:
            continue
        rate = trade_rate(state, pid, give_r)
        if p.resources[give_r] >= rate and state.bank[recv_r] > 0:
            action_id = TRADE_START + i
            if mask[action_id]:
                return action_id

    return None


def _discard_action(state: GameState, pid: int, strategy: str) -> int:
    """Discard the resource least useful to this strategy."""
    cfg = STRATEGIES[strategy]
    p   = state.players[pid]
    mask = action_mask(state)

    # Score each resource by its strategy weight (lower = discard first)
    discard_order = sorted(
        Resource,
        key=lambda r: (
            cfg["res_weights"].get(r.name.lower(), 1.0),  # prefer discarding low-weight
            p.resources[r]   # secondary: discard resource we have most of
        ),
        reverse=False  # ascending: lowest weight first
    )
    for r in discard_order:
        action_id = DISCARD_START + int(r)
        if mask[action_id]:
            return action_id

    # Fallback: any legal discard
    for action_id in range(DISCARD_START, DISCARD_START + 5):
        if mask[action_id]:
            return action_id
    raise RuntimeError("No legal discard action")


def _robber_action(state: GameState, pid: int) -> int:
    """Place robber on the hex that blocks the most opponent pip production."""
    mask  = action_mask(state)
    opp   = state.players[1 - pid]
    opp_vids = set(opp.settlements + opp.cities)

    best_score = -1.0
    best_action = None

    for h in state.board.hexes:
        action_id = ROBBER_START + h.hex_id
        if not mask[action_id]:
            continue
        # How many opponent pips does this hex block?
        pip = PIP.get(h.number_token, 0)
        opp_count = sum(
            (2 if vid in opp.cities else 1)
            for vid in state.board.hex_to_vertices[h.hex_id]
            if vid in opp_vids
        )
        score = pip * opp_count
        if score > best_score:
            best_score  = score
            best_action = action_id

    if best_action is not None:
        return best_action

    # Fallback: first legal robber hex
    for h in state.board.hexes:
        a = ROBBER_START + h.hex_id
        if mask[a]:
            return a
    raise RuntimeError("No legal robber action")


def _main_actions(
    state: GameState,
    pid: int,
    strategy: str,
    mask: np.ndarray,
    rng: np.random.Generator,
) -> int:
    cfg         = STRATEGIES[strategy]
    build_order = cfg["build_order"]
    p           = state.players[pid]

    # 1. Play knight if OWS strategy (robber control + army progress)
    if strategy == "ows" and mask[PLAY_KNIGHT]:
        return PLAY_KNIGHT

    # 2. Try builds in strategy priority order
    for build_type in build_order:
        if build_type == "place_city":
            vid = _best_city(state, pid)
            if vid is not None and can_afford(state, pid, CITY_COST):
                return CITY_START + vid

        elif build_type == "place_settlement":
            vid = _best_main_settle(state, pid, strategy)
            if vid is not None and can_afford(state, pid, SETTLEMENT_COST):
                return SETTLE_START + vid

        elif build_type == "buy_dev_card":
            if mask[BUY_DEV]:
                return BUY_DEV

        elif build_type == "place_road":
            eid = _best_main_road(state, pid, strategy)
            if eid is not None and can_afford(state, pid, ROAD_COST):
                return ROAD_START + eid

    # 3. Play road-building card if road_builder and can still place roads
    if strategy == "road_builder" and mask[PLAY_RB]:
        return PLAY_RB

    # 4. Play knight for non-OWS (still useful for robber control)
    if strategy != "ows" and mask[PLAY_KNIGHT]:
        return PLAY_KNIGHT

    # 5. Try a bank trade that advances toward the next priority build
    trade = _trade_for_build(state, pid, strategy)
    if trade is not None:
        return trade

    # 6. End turn
    if mask[END_TURN]:
        return END_TURN

    # Fallback: random legal action
    legal = np.where(mask)[0]
    return int(rng.choice(legal))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_SETUP_SETTLE_PHASES = frozenset({
    GamePhase.SETUP_P0_SETTLE_1, GamePhase.SETUP_P1_SETTLE_1,
    GamePhase.SETUP_P0_SETTLE_2, GamePhase.SETUP_P1_SETTLE_2,
})
_SETUP_ROAD_PHASES = frozenset({
    GamePhase.SETUP_P0_ROAD_1, GamePhase.SETUP_P1_ROAD_1,
    GamePhase.SETUP_P0_ROAD_2, GamePhase.SETUP_P1_ROAD_2,
})


def heuristic_action(
    state: GameState,
    strategy: str,
    rng: np.random.Generator,
) -> int:
    """
    Return the action_id the heuristic player would take.

    Parameters
    ----------
    state    : current GameState
    strategy : "road_builder" | "ows" | "balanced"
    rng      : numpy Generator for tie-breaking / fallback random choices
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. "
                         f"Choose from: {list(STRATEGIES)}")

    phase = state.phase
    pid   = state.current_player
    mask  = action_mask(state)
    legal = np.where(mask)[0]

    # Setup: settlement placement
    if phase in _SETUP_SETTLE_PHASES:
        legal_vids = legal_initial_settlement_locations(state)
        vid = _best_settle(state, strategy, legal_vids)
        return SETTLE_START + vid

    # Setup: road placement
    if phase in _SETUP_ROAD_PHASES:
        legal_eids = legal_road_locations(
            state, pid, setup_vertex=state.last_settlement_vertex
        )
        eid = _best_road_setup(state, strategy, legal_eids)
        return ROAD_START + eid

    # Roll — always roll
    if phase == GamePhase.ROLL:
        return ROLL

    # Discard — discard least useful resource
    if phase == GamePhase.DISCARD:
        return _discard_action(state, pid, strategy)

    # Robber — block opponent's best hex
    if phase == GamePhase.ROBBER:
        return _robber_action(state, pid)

    # Main actions phase
    if phase == GamePhase.ACTIONS:
        return _main_actions(state, pid, strategy, mask, rng)

    # Any other phase: random fallback
    return int(rng.choice(legal))
