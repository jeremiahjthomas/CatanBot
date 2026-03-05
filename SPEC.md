# SPEC — `balancedDie` branch

## What this branch does

Ports Colonist.io's `DiceControllerBalanced` (TypeScript) to Python as
`env/balanced_dice.py`. The engine is the authoritative dice source for
the Catan RL environment — it reproduces the exact fairness behaviour of
real Colonist.io games.

---

## Files added

| File | Description |
|---|---|
| `env/balanced_dice.py` | Balanced dice engine |
| `test_balanced_dice.py` | 24 unit tests — all passing |

---

## `env/balanced_dice.py`

### Core idea

The engine keeps a physical **deck of 36 cards** — one card per (d1, d2)
pair. Before each draw, two probability adjustments are applied on top of
the raw deck counts. When fewer than 13 cards remain the deck reshuffles
automatically before the next draw.

### Constants (match TypeScript source exactly)

| Constant | Value | Meaning |
|---|---|---|
| `MINIMUM_CARDS_BEFORE_RESHUFFLING` | 13 | Reshuffle threshold |
| `PROB_REDUCTION_RECENT` | 0.34 | Weight reduction per recent occurrence |
| `PROB_REDUCTION_STREAK` | 0.4 | Weight shift per consecutive 7 in a streak |
| `MAX_RECENT_MEMORY` | 5 | Rolling window of recent totals remembered |
| `MIN/MAX_SEVEN_ADJUSTMENT` | 0, 2 | Clamp on combined 7-weight multiplier |

### Draw sequence

For each `roll(player_id)` call:

1. **Reshuffle check** — if `cards_left < 13`, refill deck to all 36 cards.
2. **Base weights** — `weight[t] = deck_count[t] / cards_left`.
3. **Recent-roll penalty** — for each total `t`:
   `weight[t] *= max(0, 1 - recent_count[t] * 0.34)`
4. **Seven fairness** (only when `num_players >= 2`):
   `weight[7] *= clamp(imbalance_adj + streak_adj, 0, 2)`
5. **Weighted sample** — pick total proportional to adjusted weights, then
   pick a random (d1, d2) pair for that total and remove it from the deck.
6. **State update** — append to recent window, update seven counts/streak.

### Seven fairness detail

**Imbalance adjustment** (keeps cumulative 7-share fair across players):
```
imbalance_adj = 1 + (ideal_pct - player_pct) / ideal_pct
```
- `ideal_pct = 1 / num_players`
- Returns 1.0 until every player has rolled at least one 7

**Streak adjustment** (penalises a player currently on a 7 streak):
```
streak_adj = 0.4 * streak_count * (-1 if streak_player == me else +1)
```

Combined seven multiplier is clamped to [0, 2].

### Public API

```python
engine = BalancedDiceEngine(num_players=2, seed=None)

# Draw a card — returns (dice1, dice2)
d1, d2 = engine.roll(player_id)

# Peek at the probability distribution before the next draw
dist: dict[int, float] = engine.get_distribution(player_id)  # sums to 1

# Force reshuffle
engine.reshuffle()

# Read-only state (useful for RL observation vector)
engine.cards_left          # int
engine.recent_rolls        # list[int], length <= 5
engine.sevens_by_player()  # dict[player_id, count]
```

---

## `test_balanced_dice.py` — test coverage

**Standard deck** (static):
- 36 cards total
- Totals cover 2–12
- Pair counts match 2d6 (1,2,3,4,5,6,5,4,3,2,1)
- All d1, d2 in [1,6]
- d1 + d2 == total for every pair

**Engine construction**:
- Fresh engine has 36 cards and empty recent_rolls

**Drawing**:
- `roll()` returns valid (d1, d2)
- `roll()` decrements `cards_left`
- `roll()` appends to `recent_rolls`
- `recent_rolls` never exceeds `MAX_RECENT_MEMORY`

**Reshuffle**:
- Can draw down to 12 cards without a reshuffle
- Auto-reshuffle triggers when below threshold
- Manual `reshuffle()` resets `cards_left` to 36

**Long-run correctness**:
- 36,000 rolls produce totals within 15% of standard 2d6 frequencies

**`get_distribution()`**:
- Sums to 1.0
- Covers all totals 2–12
- All probabilities ≥ 0
- Still sums to 1 near the reshuffle threshold

**Adjustments**:
- Recent-roll penalty lowers probability of a recently over-rolled total
- Seven imbalance: over-represented player gets lower 7 probability
- Seven streak: streaking player gets lower 7 probability than opponent

**Reproducibility**:
- Seeded engine produces identical roll sequences
- Different seeds produce different sequences
