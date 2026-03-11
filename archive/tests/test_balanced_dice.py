"""
test_balanced_dice.py  —  unit tests for env/balanced_dice.py
Run:  python test_balanced_dice.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import sys
from collections import Counter

from env.balanced_dice import (
    BalancedDiceEngine,
    MINIMUM_CARDS_BEFORE_RESHUFFLING,
    MAX_RECENT_MEMORY,
    PROB_REDUCTION_RECENT,
    PROB_REDUCTION_STREAK,
    _STANDARD_DECK,
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


# ---------------------------------------------------------------------------
# Standard deck sanity
# ---------------------------------------------------------------------------

print("\nenv/balanced_dice.py tests\n")

check("Standard deck has 36 cards",
      sum(len(v) for v in _STANDARD_DECK.values()) == 36)

check("Standard deck totals 2-12",
      sorted(_STANDARD_DECK.keys()) == list(range(2, 13)))

expected_counts = {2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:5, 9:4, 10:3, 11:2, 12:1}
check("Standard deck pair counts correct",
      all(_STANDARD_DECK[t] and len(_STANDARD_DECK[t]) == expected_counts[t]
          for t in range(2, 13)))

check("All pairs are valid d1, d2 in [1,6]",
      all(1 <= d1 <= 6 and 1 <= d2 <= 6
          for pairs in _STANDARD_DECK.values() for d1, d2 in pairs))

check("d1 + d2 == total for every pair",
      all(d1 + d2 == total
          for total, pairs in _STANDARD_DECK.items()
          for d1, d2 in pairs))

# ---------------------------------------------------------------------------
# Deck state after construction
# ---------------------------------------------------------------------------

engine = BalancedDiceEngine(num_players=2, seed=0)

check("Fresh engine has 36 cards",
      engine.cards_left == 36)

check("Fresh engine has empty recent_rolls",
      engine.recent_rolls == [])

# ---------------------------------------------------------------------------
# Drawing cards
# ---------------------------------------------------------------------------

d1, d2 = engine.roll(player_id=0)
check("roll() returns a valid (d1,d2) pair",
      1 <= d1 <= 6 and 1 <= d2 <= 6)
check("roll() reduces cards_left by 1",
      engine.cards_left == 35)
check("roll() appends to recent_rolls",
      engine.recent_rolls == [d1 + d2])

# Draw until we have 5 recent rolls, confirm window stays at MAX_RECENT_MEMORY
engine2 = BalancedDiceEngine(num_players=2, seed=1)
for _ in range(MAX_RECENT_MEMORY + 3):
    engine2.roll(0)
check("recent_rolls never exceeds MAX_RECENT_MEMORY",
      len(engine2.recent_rolls) <= MAX_RECENT_MEMORY)

# ---------------------------------------------------------------------------
# Deck depletion and auto-reshuffle
# ---------------------------------------------------------------------------

engine3 = BalancedDiceEngine(num_players=2, seed=2)
# Draw until one card before the reshuffle threshold
target = engine3.cards_left - (MINIMUM_CARDS_BEFORE_RESHUFFLING - 1)
for _ in range(target):
    engine3.roll(0)
check(f"Can draw down to {engine3.cards_left} cards without reshuffle",
      engine3.cards_left == MINIMUM_CARDS_BEFORE_RESHUFFLING - 1)
# Next draw should trigger reshuffle (cards_left resets to 35 after one draw)
engine3.roll(0)
check("Reshuffle triggered when below threshold (cards_left resets)",
      engine3.cards_left >= MINIMUM_CARDS_BEFORE_RESHUFFLING - 1)

# Manual reshuffle
engine4 = BalancedDiceEngine(num_players=2, seed=3)
for _ in range(10):
    engine4.roll(0)
engine4.reshuffle()
check("Manual reshuffle resets cards_left to 36",
      engine4.cards_left == 36)

# ---------------------------------------------------------------------------
# Long-run distribution converges to standard 2d6 frequencies
# (the deck + reshuffle threshold means you can never exhaust all 36 cards
# in one pass, but over many rolls the distribution should match 2d6)
# ---------------------------------------------------------------------------

engine5 = BalancedDiceEngine(num_players=2, seed=42)
N = 36_000
counts = Counter()
for i in range(N):
    d1, d2 = engine5.roll(i % 2)
    counts[d1 + d2] += 1
# Each total should appear within 15% of its expected frequency
ideal = {t: expected_counts[t] / 36 for t in range(2, 13)}
within_tolerance = all(
    abs(counts[t] / N - ideal[t]) < 0.15 * ideal[t]
    for t in range(2, 13)
)
check("Long-run distribution matches standard 2d6 (36k rolls, 15% tolerance)",
      within_tolerance,
      ", ".join(f"{t}:{counts[t]/N:.3f}(exp {ideal[t]:.3f})" for t in range(2,13)))

# ---------------------------------------------------------------------------
# get_distribution sums to 1
# ---------------------------------------------------------------------------

engine6 = BalancedDiceEngine(num_players=2, seed=5)
dist = engine6.get_distribution(player_id=0)
check("get_distribution() sums to 1.0",
      abs(sum(dist.values()) - 1.0) < 1e-9)
check("get_distribution() covers all totals 2-12",
      sorted(dist.keys()) == list(range(2, 13)))
check("get_distribution() all probabilities >= 0",
      all(p >= 0 for p in dist.values()))

# get_distribution below reshuffle threshold uses full-deck base
engine7 = BalancedDiceEngine(num_players=2, seed=6)
for _ in range(36 - MINIMUM_CARDS_BEFORE_RESHUFFLING + 1):
    engine7.roll(0)
dist7 = engine7.get_distribution(player_id=0)
check("get_distribution() still sums to 1 near reshuffle threshold",
      abs(sum(dist7.values()) - 1.0) < 1e-9)

# ---------------------------------------------------------------------------
# Recent-roll penalty
# ---------------------------------------------------------------------------

engine8 = BalancedDiceEngine(num_players=2, seed=7)
# Manually inject 5 rolls of total 7 into the recent window
for _ in range(MAX_RECENT_MEMORY):
    engine8._recent_rolls.append(7)
    engine8._recent_count[7] += 1
# Weight of 7 should be reduced by (1 - 5*0.34) = clamped to 0
dist8 = engine8.get_distribution(player_id=0)
# After max memory penalty the raw seven weight could be 0 or very small
penalty = MAX_RECENT_MEMORY * PROB_REDUCTION_RECENT  # 5 * 0.34 = 1.7 → clamp to 0
check("Recent-roll penalty reduces high-frequency total's probability",
      dist8[7] < (6 / 36))  # should be lower than the unconstrained 6/36

# ---------------------------------------------------------------------------
# Seven imbalance adjustment
# ---------------------------------------------------------------------------

engine9 = BalancedDiceEngine(num_players=2, seed=8)
# Player 0 has rolled many 7s, player 1 has rolled none
engine9._total_sevens[0] = 10
engine9._total_sevens[1] = 0
dist_p0 = engine9.get_distribution(player_id=0)
dist_p1 = engine9.get_distribution(player_id=1)
check("Seven imbalance: over-represented player gets lower 7 prob",
      dist_p0[7] < dist_p1[7],
      f"p0={dist_p0[7]:.4f} p1={dist_p1[7]:.4f}")

# ---------------------------------------------------------------------------
# Seven streak adjustment
# ---------------------------------------------------------------------------

engine10 = BalancedDiceEngine(num_players=2, seed=9)
# Player 0 is on a 3-roll streak
engine10._streak_player = 0
engine10._streak_count  = 3
dist_streaker    = engine10.get_distribution(player_id=0)
dist_nonstreaker = engine10.get_distribution(player_id=1)
check("Seven streak: streaking player gets lower 7 prob",
      dist_streaker[7] < dist_nonstreaker[7],
      f"streaker={dist_streaker[7]:.4f} non-streaker={dist_nonstreaker[7]:.4f}")

# ---------------------------------------------------------------------------
# Seeded reproducibility
# ---------------------------------------------------------------------------

def roll_sequence(seed, n=20):
    e = BalancedDiceEngine(num_players=2, seed=seed)
    return [e.roll(i % 2) for i in range(n)]

check("Seeded engine is reproducible",
      roll_sequence(99) == roll_sequence(99))
check("Different seeds produce different sequences",
      roll_sequence(1) != roll_sequence(2))

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

total = passed + failed
print(f"\n{total}/{total} passed" if failed == 0
      else f"\n{passed}/{total} passed  ({failed} failed)")
sys.exit(0 if failed == 0 else 1)
