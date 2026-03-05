# SPEC — `environment` branch

## What this branch does

Adds the action space, state-transition functions, and the `CatanEnv`
Gym-compatible wrapper.  After this branch a complete 1v1 game can be
played from first setup placement through to a winner using either a
random policy or any Gym-compatible RL agent.

Also extends `env/game_state.py` with three fields needed by the
apply functions: `discard_target`, `roller`, and `roads_left_to_place`.

---

## Files added / modified

| File | Description |
|---|---|
| `env/game_state.py` | +3 fields on `GameState` |
| `env/actions.py` | Flat action space, mask, decoder |
| `env/catan_env.py` | All `apply_*` functions + `CatanEnv` |
| `test_catan_env.py` | 62 unit tests — all passing |

---

## `env/actions.py`

### Action index layout (249 total)

| Range | Action | Count |
|---|---|---|
| 0–53 | `place_settlement(vertex)` | 54 |
| 54–107 | `place_city(vertex)` | 54 |
| 108–179 | `place_road(edge)` | 72 |
| 180 | `roll_dice` | 1 |
| 181–199 | `move_robber(hex)` | 19 |
| 200 | `buy_dev_card` | 1 |
| 201 | `play_knight` | 1 |
| 202 | `play_road_building` | 1 |
| 203–217 | `play_year_of_plenty(combo)` | 15 |
| 218–222 | `play_monopoly(resource)` | 5 |
| 223–242 | `bank_trade(give, recv)` | 20 |
| 243 | `end_turn` | 1 |
| 244–248 | `discard_resource(resource)` | 5 |

**YOP combos**: 15 `(r1, r2)` pairs with repetition where `r1 ≤ r2`.

**Trade combos**: 20 `(give, recv)` ordered pairs where `give ≠ recv`.

### `action_mask(state) -> np.ndarray[bool, (249,)]`

Phase-aware boolean mask.  During `ACTIONS`, also checks:
- Affordability for build/buy actions
- Road-building sub-phase: only road placement shown while `roads_left_to_place > 0`
- Dev card timing: excludes cards bought this turn; only one per turn

### `decode_action(action_id) -> (str, param)`

Maps every index to `(action_type, param)` for use by `apply_action`.

---

## `env/catan_env.py`

### State transition functions

| Function | What it does |
|---|---|
| `apply_place_settlement` | Pays cost (if not setup), adds settlement, awards setup resources in round 2, advances setup phase |
| `apply_place_road` | Pays cost (if not free), adds road, decrements `roads_left_to_place`, advances setup phase |
| `apply_place_city` | Pays city cost, swaps settlement→city |
| `apply_roll_dice` | Rolls balanced dice; distributes resources or triggers discard/robber |
| `apply_discard` | Removes one card; advances through all discarding players then to ROBBER |
| `apply_move_robber` | Places robber; auto-steals one random card from opponent if possible |
| `apply_buy_dev_card` | Pays, draws top of deck, marks as bought-this-turn |
| `apply_play_knight` | Removes knight, increments count, enters ROBBER phase |
| `apply_play_road_building` | Removes card, sets `roads_left_to_place = 2` |
| `apply_play_year_of_plenty` | Removes card, grants 2 resources from bank |
| `apply_play_monopoly` | Removes card, steals all of one resource from opponent |
| `apply_bank_trade` | Gives N of `give_r` to bank, receives 1 of `recv_r`; rate from `trade_rate()` |
| `apply_end_turn` | Clears per-turn state, switches player, increments turn_number |
| `apply_action(state, action_id)` | Decodes and dispatches; returns winner or None |

### Discard flow

When rolling 7: if any player has > 7 cards, `players_to_discard` is
populated (roller first), `discard_target` is set, and phase → DISCARD.
Each `apply_discard` call decrements `discard_target`; when 0, moves to
next player or → ROBBER (restoring `current_player = roller`).

### Road-building sub-phase

`roads_left_to_place > 0` during ACTIONS causes `action_mask` to show
only road placement.  Each `apply_place_road` decrements the counter.
No separate phase needed.

### Observation vector: `encode_observation(state, pid) -> float32[460]`

Always encoded from the current player's perspective:

| Features | Size |
|---|---|
| Vertex occupancy (self_settle, self_city, opp_settle, opp_city) | 54×4 = 216 |
| Edge roads (self, opp) | 72×2 = 144 |
| Hex state (has_robber, token/12) | 19×2 = 38 |
| Own resources (/19) | 5 |
| Opponent total resources (/20) | 1 |
| Own dev card counts by type (/5) | 5 |
| Knights played (self/opp, /14) | 2 |
| Opponent dev hand count (/25) | 1 |
| Special cards (lr/la × 2 players) | 4 |
| Bank resources (/19) | 5 |
| Dev deck remaining (/25) | 1 |
| VP visible (self/opp, /15) | 2 |
| Last roll one-hot (2–12) | 11 |
| Next-roll distribution | 11 |
| Phase one-hot (13 phases) | 13 |
| Turn number (/100, capped) | 1 |
| **Total** | **460** |

### `CatanEnv`

```python
env = CatanEnv(seed=42)
obs, info = env.reset()       # info has "action_mask", "current_player"
obs, r, done, _, info = env.step(action_id)
legal = env.legal_actions()   # list of valid action indices
```

Reward: sparse — `+1.0` win, `-1.0` loss, `0.0` all other steps.

---

## `test_catan_env.py` — test coverage (62 tests)

- Action constants: ACTION_DIM, combo sizes, decode coverage
- `decode_action`: key action types round-trip correctly
- `reset()`: shape, dtype, mask, current_player
- `encode_observation`: shape, all-finite
- Setup phases: correct mask per phase, 54 available vertices at start
- Full setup sequence: all 8 placements → ROLL, correct structure counts
- Roll: only ROLL legal; 8 produces resource; 7 → ROBBER/DISCARD
- Discard: mask shows only discard actions; transitions correctly
- Build actions: settlement/road/city each deduct cost and update state
- `end_turn`: switches player, resets phase, increments turn
- Dev cards: buy (deck, hand, resources), knight (phase, count, hand),
  monopoly (steal), year of plenty (grant), road building (free roads)
- Bank trade: 4:1 and 3:1 port rates
- 10 random games all complete with isolated RNG (seeds 0–9)
- Win condition detected at 15 VP
