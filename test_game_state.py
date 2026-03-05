"""
test_game_state.py  —  unit tests for env/game_state.py
Run:  python test_game_state.py
"""

import sys
from copy import deepcopy

from env.game_state import (
    GameState, PlayerState, GamePhase, Resource, DevCard,
    new_game, visible_vp, total_vp, check_winner,
    can_afford, has_port, trade_rate,
    legal_initial_settlement_locations, legal_settlement_locations,
    legal_city_locations, legal_road_locations, legal_robber_hexes,
    production_for_roll, compute_road_length, update_special_cards,
    BANK_START, DEV_DECK_COMPOSITION, WIN_VP,
    LONGEST_ROAD_MIN, LARGEST_ARMY_MIN, FRIENDLY_ROBBER_MIN_VP,
    _VERTEX_ADJACENCY, _TILE_TO_RESOURCE,
)
from env.board import _EDGE_LIST

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


print("\nenv/game_state.py tests\n")

# ---------------------------------------------------------------------------
# new_game()
# ---------------------------------------------------------------------------

g = new_game(seed=42)

check("new_game: starts in SETUP_P0_SETTLE_1",
      g.phase == GamePhase.SETUP_P0_SETTLE_1)

check("new_game: current_player is 0",
      g.current_player == 0)

check("new_game: board has 19 hexes, 54 vertices, 72 edges",
      len(g.board.hexes) == 19 and len(g.board.vertices) == 54 and
      len(g.board.edges) == 72)

check("new_game: bank has 19 of each resource",
      all(g.bank[r] == 19 for r in Resource))

check("new_game: dev deck has 25 cards",
      len(g.dev_deck) == 25)

check("new_game: dev deck composition matches standard",
      sorted(g.dev_deck) == sorted(DEV_DECK_COMPOSITION))

check("new_game: both players start with empty hands",
      all(g.players[pid].resource_total() == 0 for pid in range(2)))

check("new_game: robber on desert",
      g.board.hexes[g.robber_hex].resource == "desert")

check("new_game: no winner",
      check_winner(g) is None)

check("new_game: seeded reproducibility",
      new_game(seed=99).board.hexes[0].resource ==
      new_game(seed=99).board.hexes[0].resource)

check("new_game: different seeds produce different boards",
      new_game(seed=1).board.hexes[3].resource !=
      new_game(seed=2).board.hexes[3].resource or
      new_game(seed=1).board.hexes[5].number_token !=
      new_game(seed=2).board.hexes[5].number_token)

# ---------------------------------------------------------------------------
# VP helpers
# ---------------------------------------------------------------------------

g2 = new_game(seed=1)

check("visible_vp: starts at 0",
      visible_vp(g2, 0) == 0 and visible_vp(g2, 1) == 0)

# Manually give P0 2 settlements
g2.players[0].settlements = [0, 5]
check("visible_vp: 2 settlements = 2 VP",
      visible_vp(g2, 0) == 2)

# Upgrade one to a city
g2.players[0].cities = [0]
g2.players[0].settlements = [5]
check("visible_vp: 1 settlement + 1 city = 3 VP",
      visible_vp(g2, 0) == 3)

# Longest road bonus
g2.players[0].has_longest_road = True
check("visible_vp: longest road adds 2 VP",
      visible_vp(g2, 0) == 5)

# VP dev cards are hidden
g2.players[0].dev_hand = [DevCard.VICTORY_POINT, DevCard.VICTORY_POINT]
check("visible_vp: VP dev cards NOT counted in visible_vp",
      visible_vp(g2, 0) == 5)
check("total_vp: VP dev cards ARE counted",
      total_vp(g2, 0) == 7)

# Win condition
g3 = new_game(seed=2)
g3.players[0].settlements   = list(range(5))
g3.players[0].cities        = list(range(4))
g3.players[0].has_longest_road = True
g3.players[0].has_largest_army = True
# 5 + 8 + 2 + 2 = 17 VP — wins
check("check_winner: detects winner",
      check_winner(g3) == 0)
check("check_winner: no false positive for P1",
      check_winner(g3) != 1)

# ---------------------------------------------------------------------------
# can_afford
# ---------------------------------------------------------------------------

g4 = new_game(seed=3)
g4.players[0].resources = {Resource.WOOD: 1, Resource.BRICK: 1,
                            Resource.SHEEP: 1, Resource.WHEAT: 1,
                            Resource.ORE: 0}

check("can_afford: settlement cost with exact resources",
      can_afford(g4, 0, {Resource.WOOD: 1, Resource.BRICK: 1,
                          Resource.SHEEP: 1, Resource.WHEAT: 1}))

check("can_afford: city cost without ore",
      not can_afford(g4, 0, {Resource.WHEAT: 2, Resource.ORE: 3}))

g4.players[0].resources[Resource.ORE] = 3
g4.players[0].resources[Resource.WHEAT] = 2
check("can_afford: city cost with enough resources",
      can_afford(g4, 0, {Resource.WHEAT: 2, Resource.ORE: 3}))

# ---------------------------------------------------------------------------
# has_port / trade_rate
# ---------------------------------------------------------------------------

g5 = new_game(seed=4)
# Find a vertex with a wood port and a 3:1 port
wood_port_vertex  = next((v.vertex_id for v in g5.board.vertices if v.port == "wood"),  None)
gen_port_vertex   = next((v.vertex_id for v in g5.board.vertices if v.port == "3:1"),   None)

if wood_port_vertex is not None:
    g5.players[0].settlements = [wood_port_vertex]
    check("has_port: 2:1 wood port detected",
          has_port(g5, 0, "wood"))
    check("has_port: 2:1 port not confused with 3:1",
          not has_port(g5, 0, "3:1"))
    check("trade_rate: 2 with 2:1 wood port",
          trade_rate(g5, 0, Resource.WOOD) == 2)
    check("trade_rate: 4 for resource without port",
          trade_rate(g5, 0, Resource.ORE) == 4)
else:
    print("  SKIP  (no wood port vertex found in this seed)")

if gen_port_vertex is not None:
    g5b = new_game(seed=4)
    g5b.players[0].settlements = [gen_port_vertex]
    check("trade_rate: 3 with 3:1 generic port",
          trade_rate(g5b, 0, Resource.BRICK) == 3)

# ---------------------------------------------------------------------------
# legal_initial_settlement_locations
# ---------------------------------------------------------------------------

g6 = new_game(seed=5)
locs = legal_initial_settlement_locations(g6)
check("legal_initial_settlement: all 54 vertices available at start",
      len(locs) == 54)

# Place a settlement at vertex 0; its neighbours should be removed
g6.players[0].settlements = [0]
locs2 = legal_initial_settlement_locations(g6)
adj0 = _VERTEX_ADJACENCY[0]
check("legal_initial_settlement: vertex 0 occupied -> removed",
      0 not in locs2)
check("legal_initial_settlement: neighbours of 0 -> removed",
      all(v not in locs2 for v in adj0))
check("legal_initial_settlement: non-adjacent vertices still valid",
      len(locs2) == 54 - 1 - len(adj0))

# ---------------------------------------------------------------------------
# legal_settlement_locations (main game)
# ---------------------------------------------------------------------------

g7 = new_game(seed=6)
# Build a 2-road chain: edge0 (va0-vb0) + edge1 (vb0-vc)
va0, vb0 = _EDGE_LIST[0]
edge1 = next(eid for eid, (a, b) in enumerate(_EDGE_LIST)
             if eid != 0 and (a == vb0 or b == vb0))
_, vc = _EDGE_LIST[edge1]
if _EDGE_LIST[edge1][0] == vb0:
    vc = _EDGE_LIST[edge1][1]
else:
    vc = _EDGE_LIST[edge1][0]
g7.players[0].roads = [0, edge1]
g7.players[0].settlements = [va0]
main_locs = legal_settlement_locations(g7, 0)

check("legal_settlement: empty game (no roads) returns empty",
      len(legal_settlement_locations(new_game(seed=6), 0)) == 0)
# vc is 2 edges away from va0, connected via road, not adjacent to va0
check("legal_settlement: vertex 2 roads away from settlement is candidate",
      vc in main_locs)
check("legal_settlement: occupied vertex not in candidates",
      va0 not in main_locs)

# ---------------------------------------------------------------------------
# legal_city_locations
# ---------------------------------------------------------------------------

g8 = new_game(seed=7)
g8.players[0].settlements = [3, 7]
check("legal_city: settlements are upgrade candidates",
      set(legal_city_locations(g8, 0)) == {3, 7})
check("legal_city: empty if no settlements",
      legal_city_locations(g8, 1) == [])

# At max cities, returns empty
g8.players[0].settlements = [0]
g8.players[0].cities = [1, 2, 3, 4]  # MAX_CITIES = 4
check("legal_city: empty at city limit",
      legal_city_locations(g8, 0) == [])

# ---------------------------------------------------------------------------
# legal_road_locations
# ---------------------------------------------------------------------------

g9 = new_game(seed=8)
# Setup mode: road must be adjacent to the given vertex
g9.players[0].settlements = [0]
setup_roads = legal_road_locations(g9, 0, setup_vertex=0)
# All edges that touch vertex 0
edges_at_0 = [eid for eid, (va, vb) in enumerate(_EDGE_LIST)
              if va == 0 or vb == 0]
check("legal_road setup: only edges touching settlement vertex",
      set(setup_roads) == set(edges_at_0))

# Main game: with a road at edge 0, can extend to connected vertices
g9b = new_game(seed=8)
va, vb = _EDGE_LIST[0]
g9b.players[0].settlements = [va]
g9b.players[0].roads = [0]
main_roads = legal_road_locations(g9b, 0)
check("legal_road main: road 0 placed -> can extend from both endpoints",
      len(main_roads) > 0)
check("legal_road main: already-placed road not in candidates",
      0 not in main_roads)

# Opponent settlement blocks road continuation
g9c = new_game(seed=8)
va, vb = _EDGE_LIST[0]
g9c.players[0].settlements = [va]
g9c.players[0].roads = [0]
g9c.players[1].settlements = [vb]   # opponent blocks vb
roads_blocked = legal_road_locations(g9c, 0)
# Cannot extend through vb (blocked by opponent), but can from va
roads_unblocked = legal_road_locations(g9b, 0)
check("legal_road main: opponent settlement blocks extension",
      len(roads_blocked) <= len(roads_unblocked))

# ---------------------------------------------------------------------------
# legal_robber_hexes
# ---------------------------------------------------------------------------

g10 = new_game(seed=9)
robber_hex = g10.robber_hex

# No structures placed: friendly robber with 0 VP means opponent can't be targeted
check("legal_robber: current hex excluded",
      robber_hex not in legal_robber_hexes(g10, 0))
check("legal_robber: 18 other hexes returned when opponent has no structures",
      len(legal_robber_hexes(g10, 0)) == 18)

# Give opponent >= 3 VP (place 3 settlements)
g10b = new_game(seed=9)
g10b.players[1].settlements = [0, 5, 10]
opp_hexes = set()
from env.board import _VERTEX_TO_HEXES
for vid in [0, 5, 10]:
    opp_hexes.update(_VERTEX_TO_HEXES[vid])
opp_hexes.discard(g10b.robber_hex)
robber_options = legal_robber_hexes(g10b, 0)
check("legal_robber: opponent with >=3 VP can be targeted",
      any(h in robber_options for h in opp_hexes))

# Opponent has only 2 visible VP — friendly robber blocks targeting them
g10c = new_game(seed=9)
g10c.players[1].settlements = [0, 5]   # 2 VP
opp_hexes_low = set()
for vid in [0, 5]:
    opp_hexes_low.update(_VERTEX_TO_HEXES[vid])
opp_hexes_low.discard(g10c.robber_hex)
robber_opts_low = legal_robber_hexes(g10c, 0)
check("legal_robber: friendly robber — opponent with 2 VP cannot be targeted",
      not any(h in robber_opts_low for h in opp_hexes_low))

# ---------------------------------------------------------------------------
# production_for_roll
# ---------------------------------------------------------------------------

g11 = new_game(seed=10)
# Place P0 settlement on a vertex adjacent to hex with token 6
hex6 = next((h for h in g11.board.hexes if h.number_token == 6), None)
if hex6:
    v_on_hex6 = g11.board.hex_to_vertices[hex6.hex_id][0]
    g11.players[0].settlements = [v_on_hex6]
    prod = production_for_roll(g11, 6)
    res6 = _TILE_TO_RESOURCE.get(hex6.resource)
    check("production_for_roll: settlement on hex-6 produces on roll 6",
          res6 is not None and prod.get(0, {}).get(res6, 0) == 1)

    # Robber on that hex blocks production
    g11b = deepcopy(g11)
    g11b.robber_hex = hex6.hex_id
    prod_blocked = production_for_roll(g11b, 6)
    check("production_for_roll: robber blocks production",
          prod_blocked.get(0, {}).get(res6, 0) == 0)

    # City produces 2
    g11c = deepcopy(g11)
    g11c.players[0].settlements = []
    g11c.players[0].cities = [v_on_hex6]
    prod_city = production_for_roll(g11c, 6)
    check("production_for_roll: city produces 2",
          prod_city.get(0, {}).get(res6, 0) == 2)

# Bank scarcity: if bank has 0 of the resource, no one gets it
g11d = deepcopy(g11)
if hex6:
    g11d.bank[res6] = 0
    prod_empty = production_for_roll(g11d, 6)
    check("production_for_roll: empty bank -> no production",
          prod_empty.get(0, {}).get(res6, 0) == 0)

# ---------------------------------------------------------------------------
# compute_road_length
# ---------------------------------------------------------------------------

g12 = new_game(seed=11)

check("road_length: 0 with no roads",
      compute_road_length(g12, 0) == 0)

# Build a straight line of 3 roads
g12.players[0].settlements = [0]
# edges 0, 1, 2 may not form a line; build a known chain
# chain: edge 0 connects va0-vb0, edge adjacent to vb0 connects to vc, etc.
va, vb = _EDGE_LIST[0]
# find an edge incident to vb (other than edge 0)
chain = [0]
cur = vb
for eid, (a, b) in enumerate(_EDGE_LIST):
    if eid == 0:
        continue
    if a == cur or b == cur:
        chain.append(eid)
        cur = b if a == cur else a
        if len(chain) == 3:
            break

g12.players[0].roads = chain
length = compute_road_length(g12, 0)
check(f"road_length: {len(chain)}-road chain = {len(chain)}",
      length == len(chain))

# Opponent settlement in the middle breaks the chain
if len(chain) >= 3:
    va_mid, vb_mid = _EDGE_LIST[chain[1]]
    # vb_mid is the shared vertex between chain[1] and chain[2]
    _, next_v = _EDGE_LIST[chain[2]]
    break_v = next_v if vb_mid != next_v else va_mid
    g12b = deepcopy(g12)
    g12b.players[1].settlements = [vb_mid]
    broken = compute_road_length(g12b, 0)
    check("road_length: opponent settlement breaks road",
          broken < length)

# ---------------------------------------------------------------------------
# update_special_cards
# ---------------------------------------------------------------------------

g13 = new_game(seed=12)

# No one qualifies yet
update_special_cards(g13)
check("update_special_cards: no holder with 0 roads/knights",
      g13.longest_road_holder is None and g13.largest_army_holder is None)

# Give P0 a 5-road chain (minimum for longest road)
# Build it by giving P0 roads 0-4 in a chain
g13.players[0].roads = list(range(LONGEST_ROAD_MIN))
update_special_cards(g13)
# May or may not be a valid chain — just check assignment logic
# Instead, test with a known configuration
g13b = new_game(seed=12)
# Force road length by mocking compute_road_length via monkey-patching isn't clean;
# instead give P0 exactly LONGEST_ROAD_MIN distinct edges from a connected region
# (just pick the first 5 edges which may not form a chain — use the length output)
g13b.players[0].roads = list(range(LONGEST_ROAD_MIN))
road_len = compute_road_length(g13b, 0)
update_special_cards(g13b)
if road_len >= LONGEST_ROAD_MIN:
    check("update_special_cards: P0 gets longest road when qualified",
          g13b.longest_road_holder == 0)
else:
    check("update_special_cards: no longest road when length < minimum",
          g13b.longest_road_holder is None)

# Largest army: P0 plays 3 knights
g13c = new_game(seed=12)
g13c.players[0].knights_played = LARGEST_ARMY_MIN
update_special_cards(g13c)
check("update_special_cards: P0 gets largest army at minimum threshold",
      g13c.largest_army_holder == 0)

# P1 ties P0 — holder keeps the card (tie goes to current holder)
g13c.players[1].knights_played = LARGEST_ARMY_MIN
update_special_cards(g13c)
check("update_special_cards: tie does not transfer largest army",
      g13c.largest_army_holder == 0)

# P1 exceeds P0 — transfers
g13c.players[1].knights_played = LARGEST_ARMY_MIN + 1
update_special_cards(g13c)
check("update_special_cards: larger army transfers to challenger",
      g13c.largest_army_holder == 1)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

total = passed + failed
print(f"\n{total}/{total} passed" if failed == 0
      else f"\n{passed}/{total} passed  ({failed} failed)")
sys.exit(0 if failed == 0 else 1)
