"""
test_catan_env.py  —  unit tests for env/actions.py + env/catan_env.py
Run:  python test_catan_env.py
"""

import random
import sys
from copy import deepcopy

import numpy as np

from env.game_state import (
    GamePhase, Resource, DevCard, new_game,
    visible_vp, total_vp,
    SETTLEMENT_COST, ROAD_COST, CITY_COST, DEV_CARD_COST,
)
from env.actions import (
    action_mask, decode_action, ACTION_DIM,
    SETTLE_START, CITY_START, ROAD_START, ROLL, ROBBER_START,
    BUY_DEV, PLAY_KNIGHT, PLAY_RB, YOP_START, MONO_START,
    TRADE_START, END_TURN, DISCARD_START,
    YOP_COMBOS, TRADE_COMBOS,
)
from env.catan_env import (
    CatanEnv, apply_action, encode_observation,
    apply_place_settlement, apply_place_road, apply_roll_dice,
    apply_end_turn, OBS_DIM,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        print(f"  {PASS}  {name}")
        passed += 1
    else:
        msg = f"  {FAIL}  {name}"
        if detail:
            msg += f"  ({detail})"
        print(msg)
        failed += 1


print("\nenv/actions.py + env/catan_env.py tests\n")

# ---------------------------------------------------------------------------
# Action constants sanity
# ---------------------------------------------------------------------------

check("ACTION_DIM = 249", ACTION_DIM == 249)
check("YOP_COMBOS has 15 entries", len(YOP_COMBOS) == 15)
check("TRADE_COMBOS has 20 entries", len(TRADE_COMBOS) == 20)
check("TRADE_COMBOS has no self-trades",
      all(g != r for g, r in TRADE_COMBOS))

# decode_action covers full range without raising
errors = 0
for aid in range(ACTION_DIM):
    try:
        decode_action(aid)
    except Exception:
        errors += 1
check("decode_action covers all 249 indices", errors == 0)

# decode/encode round-trip for a few key actions
check("decode ROLL", decode_action(ROLL) == ("roll_dice", None))
check("decode END_TURN", decode_action(END_TURN) == ("end_turn", None))
check("decode SETTLE_START+3", decode_action(SETTLE_START + 3) == ("place_settlement", 3))
check("decode CITY_START+7", decode_action(CITY_START + 7) == ("place_city", 7))
check("decode ROAD_START+15", decode_action(ROAD_START + 15) == ("place_road", 15))
check("decode ROBBER_START+5", decode_action(ROBBER_START + 5) == ("move_robber", 5))
check("decode DISCARD_START+2", decode_action(DISCARD_START + 2) == ("discard_resource", Resource.SHEEP))
check("decode MONO_START+1", decode_action(MONO_START + 1) == ("play_monopoly", Resource.BRICK))

# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------

env = CatanEnv(seed=0)
obs, info = env.reset()
check("reset() returns obs of correct shape", obs.shape == (OBS_DIM,))
check("reset() obs is float32", obs.dtype == np.float32)
check("reset() action_mask in info", "action_mask" in info)
check("reset() mask has at least 1 legal action", info["action_mask"].any())
check("reset() current_player is 0", info["current_player"] == 0)

import numpy as np
obs2 = encode_observation(env.state, 0)
check("encode_observation shape", obs2.shape == (OBS_DIM,))
check("encode_observation all finite", np.all(np.isfinite(obs2)))

# ---------------------------------------------------------------------------
# Action mask in each setup phase
# ---------------------------------------------------------------------------

g = new_game(seed=1)
mask = action_mask(g)
settle_actions = np.where(mask[SETTLE_START:SETTLE_START+54])[0]
check("SETUP_P0_SETTLE_1: only settlement actions legal",
      mask[SETTLE_START:SETTLE_START+54].any() and
      not mask[ROAD_START:ROAD_START+72].any() and
      not mask[ROLL])
check("SETUP_P0_SETTLE_1: 54 vertices available",
      settle_actions.shape[0] == 54)

# After placing a settlement, road phase starts
g.phase = GamePhase.SETUP_P0_ROAD_1
g.players[0].settlements = [10]
g.last_settlement_vertex = 10
mask2 = action_mask(g)
check("SETUP_P0_ROAD_1: only road actions legal",
      mask2[ROAD_START:ROAD_START+72].any() and
      not mask2[SETTLE_START:SETTLE_START+54].any())

# ---------------------------------------------------------------------------
# Full setup phase transition sequence
# ---------------------------------------------------------------------------

env2 = CatanEnv(seed=2)
obs, info = env2.reset()
phases_seen = [env2.state.phase]

# Play through all 8 setup actions (4 settlements + 4 roads)
for _ in range(8):
    legal = np.where(info["action_mask"])[0]
    check(f"Setup action available in phase {env2.state.phase.name}",
          len(legal) > 0)
    action = int(legal[0])
    obs, _, done, _, info = env2.step(action)
    phases_seen.append(env2.state.phase)
    if done:
        break

check("Setup ends in ROLL phase", env2.state.phase == GamePhase.ROLL)
check("Both players have 2 settlements after setup",
      len(env2.state.players[0].settlements) == 2 and
      len(env2.state.players[1].settlements) == 2)
check("Both players have 2 roads after setup",
      len(env2.state.players[0].roads) == 2 and
      len(env2.state.players[1].roads) == 2)

# ---------------------------------------------------------------------------
# Roll and production
# ---------------------------------------------------------------------------

g3 = new_game(seed=3)
g3.phase = GamePhase.ROLL
g3.current_player = 0
# Place settlements with known adjacencies to verify production
# Find a hex with token 8 and place a settlement there
hex8 = next((h for h in g3.board.hexes if h.number_token == 8), None)
if hex8:
    v8 = g3.board.hex_to_vertices[hex8.hex_id][0]
    g3.players[0].settlements = [v8]
    # Force a roll of 8 by injecting into dice state
    old_total = None
    for _ in range(1000):
        g4 = deepcopy(g3)
        g4.phase = GamePhase.ROLL
        mask_r = action_mask(g4)
        check("ROLL: only roll action legal",
              mask_r[ROLL] and not mask_r[END_TURN], "")
        apply_roll_dice(g4)
        if g4.last_roll == 8:
            break
    check("Rolling 8 produces resource for settlement on hex-8",
          g4.phase == GamePhase.ACTIONS,
          f"phase={g4.phase}")

# ---------------------------------------------------------------------------
# Seven triggers robber
# ---------------------------------------------------------------------------

# Roll repeatedly (advancing dice each time) until we hit a 7
g5_base = new_game(seed=4)
g6 = None
for _ in range(500):
    g6 = deepcopy(g5_base)
    g6.phase = GamePhase.ROLL
    g6.current_player = 0
    apply_roll_dice(g6)
    g5_base.dice.roll(0)   # advance rng so next deepcopy gets a different roll
    if g6.last_roll == 7:
        break
check("Roll 7 -> ROBBER or DISCARD phase",
      g6 is not None and g6.last_roll == 7 and
      g6.phase in (GamePhase.ROBBER, GamePhase.DISCARD))

# ---------------------------------------------------------------------------
# Discard mechanic
# ---------------------------------------------------------------------------

g7 = new_game(seed=5)
g7.phase = GamePhase.ROLL
g7.current_player = 0
# Give both players > 7 cards
g7.players[0].resources = {r: 2 for r in Resource}   # 10 cards
g7.players[1].resources = {r: 2 for r in Resource}   # 10 cards
# Inject a 7
g7.last_roll = 7
g7.roller    = 0
g7.players_to_discard = [0, 1]
g7.discard_target = 5   # must discard 5 (floor(10/2))
g7.phase = GamePhase.DISCARD
g7.current_player = 0

mask_d = action_mask(g7)
check("DISCARD: only discard actions legal",
      mask_d[DISCARD_START:DISCARD_START+5].any() and not mask_d[ROLL])

# Discard 5 cards
for _ in range(5):
    legal_d = np.where(mask_d)[0]
    from env.catan_env import apply_discard
    r = decode_action(int(legal_d[0]))[1]
    apply_discard(g7, g7.current_player, r)
    mask_d = action_mask(g7)
    if g7.phase != GamePhase.DISCARD or g7.current_player != 0:
        break

check("After P0 discards, moves to P1 or ROBBER",
      g7.phase in (GamePhase.DISCARD, GamePhase.ROBBER))

# ---------------------------------------------------------------------------
# Building — pay resources correctly
# ---------------------------------------------------------------------------

g8 = new_game(seed=6)
g8.phase = GamePhase.ACTIONS
g8.current_player = 0
# Give P0 enough to build a road
g8.players[0].resources = {Resource.WOOD: 2, Resource.BRICK: 2,
                             Resource.SHEEP: 0, Resource.WHEAT: 0, Resource.ORE: 0}
g8.players[0].settlements = [0]
g8.players[0].roads = [0]   # existing road

from env.catan_env import apply_bank_trade
legal_roads = np.where(action_mask(g8)[ROAD_START:ROAD_START+72])[0]
if len(legal_roads) > 0:
    eid  = int(legal_roads[0])
    pre  = g8.players[0].resources[Resource.WOOD]
    apply_action(g8, ROAD_START + eid)
    check("place_road deducts ROAD_COST",
          g8.players[0].resources[Resource.WOOD] == pre - 1)
    check("place_road adds road to player",
          eid in g8.players[0].roads)

# Settlement cost
g9 = new_game(seed=7)
g9.phase = GamePhase.ACTIONS
g9.current_player = 0
g9.players[0].resources = {Resource.WOOD: 1, Resource.BRICK: 1,
                             Resource.SHEEP: 1, Resource.WHEAT: 1, Resource.ORE: 0}
from env.board import _EDGE_LIST
va, vb = _EDGE_LIST[0]
g9.players[0].settlements = [va]
g9.players[0].roads = [0, 1]   # need road to a free vertex
# find a vertex reachable via road 1
va1, vb1 = _EDGE_LIST[1]
target_v = vb1 if va1 == vb else va1
if target_v not in g9.players[0].settlements:
    pre_wood = g9.players[0].resources[Resource.WOOD]
    apply_action(g9, SETTLE_START + target_v)
    check("place_settlement deducts SETTLEMENT_COST",
          g9.players[0].resources[Resource.WOOD] == pre_wood - 1)

# ---------------------------------------------------------------------------
# End turn advances player and phase
# ---------------------------------------------------------------------------

g10 = new_game(seed=8)
g10.phase = GamePhase.ACTIONS
g10.current_player = 0
apply_end_turn(g10)
check("end_turn switches current_player", g10.current_player == 1)
check("end_turn resets to ROLL phase", g10.phase == GamePhase.ROLL)

apply_end_turn(g10)
check("second end_turn returns to P0", g10.current_player == 0)
check("second end_turn increments turn_number", g10.turn_number == 1)

# ---------------------------------------------------------------------------
# Dev card: buy and play
# ---------------------------------------------------------------------------

g11 = new_game(seed=9)
g11.phase = GamePhase.ACTIONS
g11.current_player = 0
g11.players[0].resources = {Resource.SHEEP: 1, Resource.WHEAT: 1, Resource.ORE: 1,
                              Resource.WOOD: 0, Resource.BRICK: 0}
deck_before = len(g11.dev_deck)
apply_action(g11, BUY_DEV)
check("buy_dev_card draws from deck",
      len(g11.dev_deck) == deck_before - 1)
check("buy_dev_card adds to hand",
      len(g11.players[0].dev_hand) == 1)
check("buy_dev_card charges resources",
      g11.players[0].resources[Resource.SHEEP] == 0)

# Knight play: needs to enter ROBBER phase
g12 = new_game(seed=10)
g12.phase = GamePhase.ACTIONS
g12.current_player = 0
g12.players[0].dev_hand = [DevCard.KNIGHT]
apply_action(g12, PLAY_KNIGHT)
check("play_knight enters ROBBER phase", g12.phase == GamePhase.ROBBER)
check("play_knight increments knights_played",
      g12.players[0].knights_played == 1)
check("play_knight removes card from hand",
      DevCard.KNIGHT not in g12.players[0].dev_hand)

# Monopoly
g13 = new_game(seed=11)
g13.phase = GamePhase.ACTIONS
g13.current_player = 0
g13.players[0].dev_hand = [DevCard.MONOPOLY]
g13.players[1].resources[Resource.WOOD] = 5
pre_wood = g13.players[0].resources[Resource.WOOD]
apply_action(g13, MONO_START + int(Resource.WOOD))
check("play_monopoly steals all of resource from opponent",
      g13.players[0].resources[Resource.WOOD] == pre_wood + 5)
check("play_monopoly empties opponent",
      g13.players[1].resources[Resource.WOOD] == 0)

# Year of plenty
g14 = new_game(seed=12)
g14.phase = GamePhase.ACTIONS
g14.current_player = 0
g14.players[0].dev_hand = [DevCard.YEAR_OF_PLENTY]
pre_ore  = g14.players[0].resources[Resource.ORE]
pre_wheat = g14.players[0].resources[Resource.WHEAT]
# Combo: (ORE, WHEAT) — find its index
yop_idx = YOP_COMBOS.index((Resource.WHEAT, Resource.ORE))
apply_action(g14, YOP_START + yop_idx)
check("play_year_of_plenty grants 2 resources",
      g14.players[0].resources[Resource.ORE]   == pre_ore  + 1 and
      g14.players[0].resources[Resource.WHEAT] == pre_wheat + 1)

# ---------------------------------------------------------------------------
# Bank trade
# ---------------------------------------------------------------------------

g15 = new_game(seed=13)
g15.phase = GamePhase.ACTIONS
g15.current_player = 0
g15.players[0].resources[Resource.WOOD] = 4
pre_ore2 = g15.players[0].resources[Resource.ORE]
trade_idx = TRADE_COMBOS.index((Resource.WOOD, Resource.ORE))
apply_action(g15, TRADE_START + trade_idx)
check("bank_trade at 4:1 deducts 4 wood",
      g15.players[0].resources[Resource.WOOD] == 0)
check("bank_trade at 4:1 gives 1 ore",
      g15.players[0].resources[Resource.ORE] == pre_ore2 + 1)

# With 3:1 port
g16 = new_game(seed=14)
g16.phase = GamePhase.ACTIONS
g16.current_player = 0
port3_v = next((v.vertex_id for v in g16.board.vertices if v.port == "3:1"), None)
if port3_v is not None:
    g16.players[0].settlements = [port3_v]
    g16.players[0].resources[Resource.WOOD] = 3
    pre_ore3 = g16.players[0].resources[Resource.ORE]
    trade_idx2 = TRADE_COMBOS.index((Resource.WOOD, Resource.ORE))
    apply_action(g16, TRADE_START + trade_idx2)
    check("bank_trade at 3:1 port deducts 3 wood",
          g16.players[0].resources[Resource.WOOD] == 0)
    check("bank_trade at 3:1 port gives 1 ore",
          g16.players[0].resources[Resource.ORE] == pre_ore3 + 1)

# ---------------------------------------------------------------------------
# Full random game
# ---------------------------------------------------------------------------

def play_random_game(seed=None, max_steps=20_000):
    rng = random.Random(seed)   # isolated RNG so test order doesn't matter
    env_r = CatanEnv(seed=seed)
    obs, info = env_r.reset()
    for step in range(max_steps):
        legal = list(np.where(info["action_mask"])[0])
        if not legal:
            return None, step
        action = rng.choice(legal)
        obs, reward, done, _, info = env_r.step(action)
        if done:
            return env_r.state.winner, step
    return None, max_steps  # timed out


winners = []
for seed in range(10):
    winner, steps = play_random_game(seed=seed, max_steps=10_000)
    winners.append(winner)

check("10 random games all complete (winner determined)",
      all(w is not None for w in winners),
      str(winners))
check("Winners are player 0 or 1",
      all(w in (0, 1) for w in winners if w is not None))

# ---------------------------------------------------------------------------
# Win detection
# ---------------------------------------------------------------------------

g17 = new_game(seed=15)
g17.phase = GamePhase.ACTIONS
g17.current_player = 0
# Give P0 14 VP (just short of winning)
g17.players[0].settlements = [0]
g17.players[0].cities      = [1, 2, 3, 4]   # 1 + 8 = 9 VP
g17.players[0].has_longest_road = True        # +2 = 11 VP
g17.players[0].has_largest_army = True        # +2 = 13 VP
# 2 VP dev cards needed for 15
g17.players[0].dev_hand = [DevCard.VICTORY_POINT, DevCard.VICTORY_POINT]
from env.game_state import check_winner
check("check_winner detects 15 VP win",
      check_winner(g17) == 0,
      f"VP={total_vp(g17, 0)}")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

total = passed + failed
print(f"\n{total}/{total} passed" if failed == 0
      else f"\n{passed}/{total} passed  ({failed} failed)")
sys.exit(0 if failed == 0 else 1)
