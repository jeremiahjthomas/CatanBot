"""
env/catan_env.py

State-transition functions (apply_*) and Gym-compatible CatanEnv wrapper
for 1v1 Catan with Colonist.io rules.

The environment advances state for both players.  The caller supplies the
action for whoever is state.current_player.  For self-play the caller
routes each player's observation to the appropriate agent.

Observation vector: float32 of shape (OBS_DIM,) = (460,)
  See encode_observation() for full breakdown.

Action space: Discrete(ACTION_DIM=249)
  See env/actions.py for the index layout.

Reward: sparse  +1.0 = win, -1.0 = loss, 0.0 everywhere else.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from env.game_state import (
    GameState, PlayerState, GamePhase, Resource, DevCard,
    new_game, visible_vp, total_vp, check_winner,
    can_afford, trade_rate, production_for_roll, update_special_cards,
    legal_robber_hexes,
    ROAD_COST, SETTLEMENT_COST, CITY_COST, DEV_CARD_COST,
    WIN_VP, _VERTEX_TO_HEXES, _TILE_TO_RESOURCE,
)
from env.actions import (
    action_mask, decode_action, ACTION_DIM,
    YOP_COMBOS, TRADE_COMBOS,
)
from env.board import _EDGE_LIST

OBS_DIM = 460


# ---------------------------------------------------------------------------
# Setup phase helpers
# ---------------------------------------------------------------------------

_SETUP_SETTLE_PHASES = frozenset({
    GamePhase.SETUP_P0_SETTLE_1, GamePhase.SETUP_P1_SETTLE_1,
    GamePhase.SETUP_P0_SETTLE_2, GamePhase.SETUP_P1_SETTLE_2,
})
_SETUP_ROAD_PHASES = frozenset({
    GamePhase.SETUP_P0_ROAD_1, GamePhase.SETUP_P1_ROAD_1,
    GamePhase.SETUP_P0_ROAD_2, GamePhase.SETUP_P1_ROAD_2,
})
_ALL_SETUP_PHASES = _SETUP_SETTLE_PHASES | _SETUP_ROAD_PHASES

_SETTLE_TO_ROAD: Dict[GamePhase, GamePhase] = {
    GamePhase.SETUP_P0_SETTLE_1: GamePhase.SETUP_P0_ROAD_1,
    GamePhase.SETUP_P1_SETTLE_1: GamePhase.SETUP_P1_ROAD_1,
    GamePhase.SETUP_P1_SETTLE_2: GamePhase.SETUP_P1_ROAD_2,
    GamePhase.SETUP_P0_SETTLE_2: GamePhase.SETUP_P0_ROAD_2,
}
# road phase -> (next phase, next current_player)
_ROAD_TRANSITIONS: Dict[GamePhase, Tuple[GamePhase, int]] = {
    GamePhase.SETUP_P0_ROAD_1: (GamePhase.SETUP_P1_SETTLE_1, 1),
    GamePhase.SETUP_P1_ROAD_1: (GamePhase.SETUP_P1_SETTLE_2, 1),
    GamePhase.SETUP_P1_ROAD_2: (GamePhase.SETUP_P0_SETTLE_2, 0),
    GamePhase.SETUP_P0_ROAD_2: (GamePhase.ROLL,              0),
}


# ---------------------------------------------------------------------------
# State transition functions
# ---------------------------------------------------------------------------

def apply_place_settlement(state: GameState, pid: int,
                            vertex_id: int, is_setup: bool) -> None:
    p = state.players[pid]

    if not is_setup:
        _pay(state, pid, SETTLEMENT_COST)

    p.settlements.append(vertex_id)
    state.last_settlement_vertex = vertex_id

    # Second setup round: receive resources from all adjacent non-desert hexes
    if is_setup and state.phase in (GamePhase.SETUP_P0_SETTLE_2,
                                     GamePhase.SETUP_P1_SETTLE_2):
        for hid in state.board.vertices[vertex_id].adjacent_hexes:
            h   = state.board.hexes[hid]
            res = _TILE_TO_RESOURCE.get(h.resource)
            if res is not None and state.bank[res] > 0:
                p.resources[res]  += 1
                state.bank[res]   -= 1

    update_special_cards(state)

    if is_setup:
        state.phase = _SETTLE_TO_ROAD[state.phase]


def apply_place_road(state: GameState, pid: int,
                     edge_id: int, is_setup: bool) -> None:
    if not is_setup and state.roads_left_to_place == 0:
        _pay(state, pid, ROAD_COST)

    state.players[pid].roads.append(edge_id)
    update_special_cards(state)

    if is_setup:
        next_phase, next_player = _ROAD_TRANSITIONS[state.phase]
        state.phase          = next_phase
        state.current_player = next_player
    elif state.roads_left_to_place > 0:
        state.roads_left_to_place -= 1


def apply_place_city(state: GameState, pid: int, vertex_id: int) -> None:
    _pay(state, pid, CITY_COST)
    p = state.players[pid]
    p.settlements.remove(vertex_id)
    p.cities.append(vertex_id)
    update_special_cards(state)


def apply_roll_dice(state: GameState) -> int:
    """Roll, distribute resources or trigger robber. Returns total."""
    pid     = state.current_player
    d1, d2  = state.dice.roll(pid)
    total   = d1 + d2
    state.last_roll = total
    state.roller    = pid

    if total == 7:
        need_discard = [p.player_id for p in state.players
                        if p.resource_total() > 7]
        # Roller discards first; opponent second
        need_discard.sort(key=lambda x: 0 if x == pid else 1)
        state.players_to_discard = need_discard

        if need_discard:
            first = need_discard[0]
            state.current_player  = first
            state.discard_target  = math.floor(
                state.players[first].resource_total() / 2
            )
            state.phase = GamePhase.DISCARD
        else:
            state.phase = GamePhase.ROBBER
    else:
        prod = production_for_roll(state, total)
        for pid_recv, resources in prod.items():
            for res, amt in resources.items():
                state.players[pid_recv].resources[res] += amt
                state.bank[res]                        -= amt
        state.phase = GamePhase.ACTIONS

    return total


def apply_discard(state: GameState, pid: int, resource: Resource) -> None:
    """Discard one card; advance phase when all discards are done."""
    p = state.players[pid]
    p.resources[resource] -= 1
    state.bank[resource]  += 1
    state.discard_target  -= 1

    if state.discard_target <= 0:
        state.players_to_discard.remove(pid)
        if state.players_to_discard:
            nxt                  = state.players_to_discard[0]
            state.current_player = nxt
            state.discard_target = math.floor(
                state.players[nxt].resource_total() / 2
            )
        else:
            state.current_player = state.roller
            state.phase          = GamePhase.ROBBER


def apply_move_robber(state: GameState, pid: int, hex_id: int) -> None:
    """Place robber; auto-steal one random card from opponent if possible."""
    state.robber_hex        = hex_id
    state.board.robber_hex  = hex_id

    opponent = 1 - pid
    opp      = state.players[opponent]
    opp_vids = set(opp.settlements + opp.cities)

    if (any(vid in opp_vids for vid in state.board.hex_to_vertices[hex_id])
            and opp.resource_total() > 0):
        pool   = [r for r in Resource for _ in range(opp.resources[r])]
        stolen = state.rng.choice(pool)
        opp.resources[stolen]              -= 1
        state.players[pid].resources[stolen] += 1

    state.phase = GamePhase.ACTIONS


def apply_buy_dev_card(state: GameState, pid: int) -> None:
    _pay(state, pid, DEV_CARD_COST)
    card = state.dev_deck.pop()
    p    = state.players[pid]
    p.dev_hand.append(card)
    p.dev_bought_this_turn.append(card)


def apply_play_knight(state: GameState, pid: int) -> None:
    p = state.players[pid]
    p.dev_hand.remove(DevCard.KNIGHT)
    p.knights_played              += 1
    state.dev_card_played_this_turn = True
    state.roller                    = pid   # knight acts like rolling a 7
    update_special_cards(state)
    state.phase = GamePhase.ROBBER


def apply_play_road_building(state: GameState, pid: int) -> None:
    p = state.players[pid]
    p.dev_hand.remove(DevCard.ROAD_BUILDING)
    state.dev_card_played_this_turn = True
    # Grant up to 2 free roads (limited by remaining road pieces)
    state.roads_left_to_place = min(2, 15 - len(p.roads))


def apply_play_year_of_plenty(state: GameState, pid: int,
                               r1: Resource, r2: Resource) -> None:
    p = state.players[pid]
    p.dev_hand.remove(DevCard.YEAR_OF_PLENTY)
    state.dev_card_played_this_turn = True
    for r in (r1, r2):
        if state.bank[r] > 0:
            p.resources[r] += 1
            state.bank[r]  -= 1


def apply_play_monopoly(state: GameState, pid: int,
                        resource: Resource) -> None:
    p = state.players[pid]
    p.dev_hand.remove(DevCard.MONOPOLY)
    state.dev_card_played_this_turn = True
    opp                  = state.players[1 - pid]
    stolen               = opp.resources[resource]
    opp.resources[resource]  = 0
    p.resources[resource]   += stolen


def apply_bank_trade(state: GameState, pid: int,
                     give_r: Resource, recv_r: Resource) -> None:
    p    = state.players[pid]
    rate = trade_rate(state, pid, give_r)
    p.resources[give_r]  -= rate
    state.bank[give_r]   += rate
    p.resources[recv_r]  += 1
    state.bank[recv_r]   -= 1


def apply_end_turn(state: GameState) -> None:
    pid = state.current_player
    state.players[pid].dev_bought_this_turn.clear()
    state.dev_card_played_this_turn = False
    state.roads_left_to_place       = 0
    state.last_roll                 = None

    state.current_player = 1 - pid
    if state.current_player == 0:
        state.turn_number += 1
    state.phase = GamePhase.ROLL


def apply_action(state: GameState, action_id: int) -> Optional[int]:
    """
    Decode and apply action_id to state (mutates in place).
    Returns the winning player_id if the game just ended, else None.
    """
    action_type, param = decode_action(action_id)
    pid                = state.current_player
    is_setup           = state.phase in _ALL_SETUP_PHASES

    if action_type == "place_settlement":
        apply_place_settlement(state, pid, param, is_setup)
    elif action_type == "place_city":
        apply_place_city(state, pid, param)
    elif action_type == "place_road":
        apply_place_road(state, pid, param, is_setup)
    elif action_type == "roll_dice":
        apply_roll_dice(state)
    elif action_type == "move_robber":
        apply_move_robber(state, pid, param)
    elif action_type == "discard_resource":
        apply_discard(state, pid, param)
    elif action_type == "buy_dev_card":
        apply_buy_dev_card(state, pid)
    elif action_type == "play_knight":
        apply_play_knight(state, pid)
    elif action_type == "play_road_building":
        apply_play_road_building(state, pid)
    elif action_type == "play_year_of_plenty":
        apply_play_year_of_plenty(state, pid, *param)
    elif action_type == "play_monopoly":
        apply_play_monopoly(state, pid, param)
    elif action_type == "bank_trade":
        apply_bank_trade(state, pid, *param)
    elif action_type == "end_turn":
        apply_end_turn(state)

    winner = check_winner(state)
    if winner is not None:
        state.phase  = GamePhase.GAME_OVER
        state.winner = winner
    return winner


# ---------------------------------------------------------------------------
# Observation encoding (OBS_DIM = 460)
# ---------------------------------------------------------------------------

def encode_observation(state: GameState, pid: int) -> np.ndarray:
    """
    Encode state from player pid's perspective into a float32 vector.

    Layout (total 460):
      54*4 = 216  vertex occupancy  (self_settle, self_city, opp_settle, opp_city)
      72*2 = 144  edge roads        (self_road, opp_road)
      19*2 =  38  hex state         (has_robber, token/12)
           5       own resources    (normalised /19)
           1       opp resource total (normalised /20)
           5       own dev card counts by type (/5)
           2       knights played   (self/opp, /14)
           1       opp dev hand size (/25)
           4       special cards    (self_lr, self_la, opp_lr, opp_la)
           5       bank resources   (/19)
           1       dev deck size    (/25)
           2       VP               (self/opp, /WIN_VP)
          11       last roll one-hot (2-12)
          11       next-roll distribution
          13       phase one-hot
           1       turn number      (/100, capped)
    """
    obs    = []
    p_self = state.players[pid]
    p_opp  = state.players[1 - pid]

    s_s = set(p_self.settlements);  s_c = set(p_self.cities)
    o_s = set(p_opp.settlements);   o_c = set(p_opp.cities)
    s_r = set(p_self.roads);        o_r = set(p_opp.roads)

    # 1. Vertices (216)
    for vid in range(54):
        obs += [float(vid in s_s), float(vid in s_c),
                float(vid in o_s), float(vid in o_c)]

    # 2. Edges (144)
    for eid in range(72):
        obs += [float(eid in s_r), float(eid in o_r)]

    # 3. Hexes (38)
    for hid in range(19):
        h = state.board.hexes[hid]
        obs += [float(hid == state.robber_hex), h.number_token / 12.0]

    # 4. Own resources (5)
    for r in Resource:
        obs.append(p_self.resources[r] / 19.0)

    # 5. Opponent total resources (1)
    obs.append(p_opp.resource_total() / 20.0)

    # 6. Own dev hand by type (5)
    for dc in DevCard:
        obs.append(sum(1 for c in p_self.dev_hand if c == dc) / 5.0)

    # 7. Knights played (2)
    obs.append(p_self.knights_played / 14.0)
    obs.append(p_opp.knights_played  / 14.0)

    # 8. Opponent dev hand count (1)
    obs.append(len(p_opp.dev_hand) / 25.0)

    # 9. Special cards (4)
    obs += [float(p_self.has_longest_road), float(p_self.has_largest_army),
            float(p_opp.has_longest_road),  float(p_opp.has_largest_army)]

    # 10. Bank (5)
    for r in Resource:
        obs.append(state.bank[r] / 19.0)

    # 11. Dev deck (1)
    obs.append(len(state.dev_deck) / 25.0)

    # 12. VP (2)
    obs.append(visible_vp(state, pid)     / WIN_VP)
    obs.append(visible_vp(state, 1 - pid) / WIN_VP)

    # 13. Last roll one-hot (11)
    roll_vec = [0.0] * 11
    if state.last_roll is not None:
        roll_vec[state.last_roll - 2] = 1.0
    obs += roll_vec

    # 14. Next-roll distribution (11)
    dist = state.dice.get_distribution(pid)
    for t in range(2, 13):
        obs.append(dist[t])

    # 15. Phase one-hot (13)
    pv = [0.0] * 13
    pv[int(state.phase)] = 1.0
    obs += pv

    # 16. Turn number (1)
    obs.append(min(state.turn_number / 100.0, 1.0))

    assert len(obs) == OBS_DIM, f"obs length {len(obs)} != {OBS_DIM}"
    return np.array(obs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Gym-compatible environment wrapper
# ---------------------------------------------------------------------------

class CatanEnv:
    """
    1v1 Catan environment.

    Compatible with gymnasium's Env interface; works standalone if
    gymnasium is not installed.

    The env steps for whichever player is current_player.  For self-play,
    the training loop reads info["current_player"] to route the observation
    and action to the correct agent.
    """

    observation_space_shape = (OBS_DIM,)
    action_space_size        = ACTION_DIM

    def __init__(self, seed: Optional[int] = None):
        self._seed: Optional[int] = seed
        self.state: Optional[GameState] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        s          = seed if seed is not None else self._seed
        self.state = new_game(seed=s)
        pid        = self.state.current_player
        obs        = encode_observation(self.state, pid)
        return obs, {"action_mask": action_mask(self.state),
                     "current_player": pid}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        assert self.state is not None, "Call reset() before step()"
        pid    = self.state.current_player
        winner = apply_action(self.state, action)

        if winner is not None or self.state.phase == GamePhase.GAME_OVER:
            # Game over
            reward = 1.0 if winner == 0 else -1.0
            obs    = encode_observation(self.state, 0)
            return obs, reward, True, False, {
                "winner": winner,
                "action_mask": np.zeros(ACTION_DIM, dtype=bool),
            }

        new_pid = self.state.current_player
        obs     = encode_observation(self.state, new_pid)
        return obs, 0.0, False, False, {
            "action_mask": action_mask(self.state),
            "current_player": new_pid,
        }

    def legal_actions(self) -> List[int]:
        """Indices of currently legal actions."""
        return list(np.where(action_mask(self.state))[0])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pay(state: GameState, pid: int,
         cost: Dict[Resource, int]) -> None:
    """Deduct cost from player's hand and return to bank."""
    p = state.players[pid]
    for res, amt in cost.items():
        p.resources[res] -= amt
        state.bank[res]  += amt
