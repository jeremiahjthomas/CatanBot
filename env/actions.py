"""
env/actions.py

Flat discrete action space for the 1v1 Catan RL environment.

Action index layout (249 total):
  0 –  53  place_settlement(vertex)       54
 54 – 107  place_city(vertex)             54
108 – 179  place_road(edge)               72
180        roll_dice                       1
181 – 199  move_robber(hex)               19
200        buy_dev_card                    1
201        play_knight                     1
202        play_road_building              1
203 – 217  play_year_of_plenty(combo)     15
218 – 222  play_monopoly(resource)         5
223 – 242  bank_trade(give, recv)         20
243        end_turn                        1
244 – 248  discard_resource(resource)      5
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from env.game_state import (
    GameState, GamePhase, Resource, DevCard,
    legal_initial_settlement_locations, legal_settlement_locations,
    legal_city_locations, legal_road_locations, legal_robber_hexes,
    can_afford, trade_rate,
    ROAD_COST, SETTLEMENT_COST, CITY_COST, DEV_CARD_COST,
)

# ---------------------------------------------------------------------------
# Index constants
# ---------------------------------------------------------------------------

SETTLE_START  = 0
CITY_START    = 54
ROAD_START    = 108
ROLL          = 180
ROBBER_START  = 181
BUY_DEV       = 200
PLAY_KNIGHT   = 201
PLAY_RB       = 202
YOP_START     = 203
MONO_START    = 218
TRADE_START   = 223
END_TURN      = 243
DISCARD_START = 244
ACTION_DIM    = 249

# ---------------------------------------------------------------------------
# Fixed combo tables
# ---------------------------------------------------------------------------

# Year of plenty: 15 (r1, r2) with repetition, r1 <= r2
YOP_COMBOS: List[Tuple[Resource, Resource]] = [
    (r1, r2)
    for i, r1 in enumerate(list(Resource))
    for r2 in list(Resource)[i:]
]
assert len(YOP_COMBOS) == 15

# Bank trade: 20 (give, recv) ordered pairs, give != recv
TRADE_COMBOS: List[Tuple[Resource, Resource]] = [
    (g, r) for g in Resource for r in Resource if g != r
]
assert len(TRADE_COMBOS) == 20


# ---------------------------------------------------------------------------
# Action mask
# ---------------------------------------------------------------------------

def action_mask(state: GameState) -> np.ndarray:
    """
    Return a boolean array of shape (ACTION_DIM,).
    True  = legal action in the current state.
    """
    mask  = np.zeros(ACTION_DIM, dtype=bool)
    phase = state.phase
    pid   = state.current_player
    p     = state.players[pid]

    # --- Setup settlement ---
    if phase in (GamePhase.SETUP_P0_SETTLE_1, GamePhase.SETUP_P1_SETTLE_1,
                 GamePhase.SETUP_P0_SETTLE_2, GamePhase.SETUP_P1_SETTLE_2):
        for v in legal_initial_settlement_locations(state):
            mask[SETTLE_START + v] = True

    # --- Setup road ---
    elif phase in (GamePhase.SETUP_P0_ROAD_1, GamePhase.SETUP_P1_ROAD_1,
                   GamePhase.SETUP_P0_ROAD_2, GamePhase.SETUP_P1_ROAD_2):
        for e in legal_road_locations(state, pid,
                                       setup_vertex=state.last_settlement_vertex):
            mask[ROAD_START + e] = True

    # --- Roll ---
    elif phase == GamePhase.ROLL:
        mask[ROLL] = True

    # --- Discard (pick one card at a time) ---
    elif phase == GamePhase.DISCARD:
        for r in Resource:
            if p.resources[r] > 0:
                mask[DISCARD_START + int(r)] = True

    # --- Move robber ---
    elif phase == GamePhase.ROBBER:
        for h in legal_robber_hexes(state, pid):
            mask[ROBBER_START + h] = True

    # --- Main actions ---
    elif phase == GamePhase.ACTIONS:
        # Road-building sub-phase: only road placement allowed
        if state.roads_left_to_place > 0:
            for e in legal_road_locations(state, pid):
                mask[ROAD_START + e] = True
            return mask

        # Build settlement
        if can_afford(state, pid, SETTLEMENT_COST):
            for v in legal_settlement_locations(state, pid):
                mask[SETTLE_START + v] = True

        # Build city
        if can_afford(state, pid, CITY_COST):
            for v in legal_city_locations(state, pid):
                mask[CITY_START + v] = True

        # Build road
        if can_afford(state, pid, ROAD_COST):
            for e in legal_road_locations(state, pid):
                mask[ROAD_START + e] = True

        # Buy dev card
        if can_afford(state, pid, DEV_CARD_COST) and state.dev_deck:
            mask[BUY_DEV] = True

        # Play dev card (one per turn; not if bought this turn)
        if not state.dev_card_played_this_turn:
            playable = p.playable_dev_cards()

            if DevCard.KNIGHT in playable:
                mask[PLAY_KNIGHT] = True

            if DevCard.ROAD_BUILDING in playable:
                if legal_road_locations(state, pid):
                    mask[PLAY_RB] = True

            if DevCard.YEAR_OF_PLENTY in playable:
                for i, (r1, r2) in enumerate(YOP_COMBOS):
                    need = 2 if r1 == r2 else 1
                    if state.bank[r1] >= need and (r1 == r2 or state.bank[r2] >= 1):
                        mask[YOP_START + i] = True

            if DevCard.MONOPOLY in playable:
                for r in Resource:
                    mask[MONO_START + int(r)] = True

        # Bank trade
        for i, (give_r, recv_r) in enumerate(TRADE_COMBOS):
            rate = trade_rate(state, pid, give_r)
            if p.resources[give_r] >= rate and state.bank[recv_r] > 0:
                mask[TRADE_START + i] = True

        # End turn always legal during ACTIONS
        mask[END_TURN] = True

    return mask


# ---------------------------------------------------------------------------
# Action decoder
# ---------------------------------------------------------------------------

def decode_action(action_id: int) -> Tuple[str, object]:
    """
    Return (action_type: str, param) for the given action index.
    param is None | int | Resource | Tuple[Resource, Resource].
    """
    if SETTLE_START <= action_id < CITY_START:
        return ("place_settlement", action_id - SETTLE_START)
    if CITY_START <= action_id < ROAD_START:
        return ("place_city", action_id - CITY_START)
    if ROAD_START <= action_id < ROLL:
        return ("place_road", action_id - ROAD_START)
    if action_id == ROLL:
        return ("roll_dice", None)
    if ROBBER_START <= action_id < BUY_DEV:
        return ("move_robber", action_id - ROBBER_START)
    if action_id == BUY_DEV:
        return ("buy_dev_card", None)
    if action_id == PLAY_KNIGHT:
        return ("play_knight", None)
    if action_id == PLAY_RB:
        return ("play_road_building", None)
    if YOP_START <= action_id < MONO_START:
        return ("play_year_of_plenty", YOP_COMBOS[action_id - YOP_START])
    if MONO_START <= action_id < TRADE_START:
        return ("play_monopoly", Resource(action_id - MONO_START))
    if TRADE_START <= action_id < END_TURN:
        return ("bank_trade", TRADE_COMBOS[action_id - TRADE_START])
    if action_id == END_TURN:
        return ("end_turn", None)
    if DISCARD_START <= action_id < ACTION_DIM:
        return ("discard_resource", Resource(action_id - DISCARD_START))
    raise ValueError(f"Invalid action_id: {action_id}")
