"""
env/balanced_dice.py

Python port of Colonist.io's DiceControllerBalanced.

The engine keeps a deck of 36 cards (all 36 (d1,d2) pairs).
Before each draw, two adjustments are applied to the raw deck
probabilities:

  1. Recent-roll penalty
     Totals that appeared in the last 5 rolls have their weight
     multiplied by (1 - 0.34 * recent_count).

  2. Seven fairness (per active player)
     The 7-weight is scaled by clamp(imbalance_adj + streak_adj, 0, 2)
     to keep 7s distributed evenly across players.

When fewer than 13 cards remain, the deck reshuffles before the next draw.
"""

from __future__ import annotations

import random
from typing import Optional


# ---------------------------------------------------------------------------
# Constants — match the TypeScript source exactly
# ---------------------------------------------------------------------------

MINIMUM_CARDS_BEFORE_RESHUFFLING = 13
PROB_REDUCTION_RECENT            = 0.34   # weight reduction per recent occurrence
PROB_REDUCTION_STREAK            = 0.4    # weight adjustment per streak roll
MAX_RECENT_MEMORY                = 5
MAX_SEVEN_ADJUSTMENT             = 2.0
MIN_SEVEN_ADJUSTMENT             = 0.0

# All 36 (d1, d2) pairs grouped by total — never mutated
_STANDARD_DECK: dict[int, list[tuple[int, int]]] = {
    2:  [(1,1)],
    3:  [(1,2),(2,1)],
    4:  [(1,3),(2,2),(3,1)],
    5:  [(1,4),(2,3),(3,2),(4,1)],
    6:  [(1,5),(2,4),(3,3),(4,2),(5,1)],
    7:  [(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)],
    8:  [(2,6),(3,5),(4,4),(5,3),(6,2)],
    9:  [(3,6),(4,5),(5,4),(6,3)],
    10: [(4,6),(5,5),(6,4)],
    11: [(5,6),(6,5)],
    12: [(6,6)],
}


class BalancedDiceEngine:
    """
    Replicates Colonist.io's DiceControllerBalanced for 1v1 (or N-player) games.

    Usage
    -----
        engine = BalancedDiceEngine(num_players=2)
        d1, d2 = engine.roll(player_id=0)           # draw a card
        dist   = engine.get_distribution(player_id=0)  # peek at next probabilities
    """

    def __init__(self, num_players: int = 2, seed: Optional[int] = None):
        self._rng          = random.Random(seed)
        self._num_players  = num_players

        # Live deck: total -> list of remaining (d1, d2) pairs
        self._deck:       dict[int, list[tuple[int, int]]] = {}
        self._cards_left: int = 0

        # Recent-roll window
        self._recent_rolls: list[int]      = []
        self._recent_count: dict[int, int] = {t: 0 for t in range(2, 13)}

        # Per-player seven tracking
        self._total_sevens: dict[int, int] = {p: 0 for p in range(num_players)}

        # Streak state
        self._streak_player: Optional[int] = None
        self._streak_count:  int           = 0

        self._reshuffle()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def roll(self, player_id: int) -> tuple[int, int]:
        """
        Draw one card for player_id. Automatically reshuffles if the deck
        drops below MINIMUM_CARDS_BEFORE_RESHUFFLING before drawing.
        Returns (dice1, dice2).
        """
        if self._cards_left < MINIMUM_CARDS_BEFORE_RESHUFFLING:
            self._reshuffle()

        weights = self._compute_weights(player_id)
        total   = self._weighted_choice(weights)

        pair = self._rng.choice(self._deck[total])
        self._deck[total].remove(pair)
        self._cards_left -= 1

        self._update_recent(total)
        if total == 7:
            self._update_sevens(player_id)

        return pair

    def get_distribution(self, player_id: int) -> dict[int, float]:
        """
        Return the probability distribution that would apply on the next
        roll for player_id, without drawing.  Values sum to 1.0.

        If the deck is below the reshuffle threshold, the distribution is
        computed against a fresh full deck (matching what roll() would do).
        """
        if self._cards_left < MINIMUM_CARDS_BEFORE_RESHUFFLING:
            base = {t: len(_STANDARD_DECK[t]) / 36 for t in range(2, 13)}
        else:
            base = {t: len(self._deck[t]) / self._cards_left
                    for t in range(2, 13)}

        weights = self._apply_adjustments(base, player_id)
        total_w = sum(weights.values())
        if total_w <= 0:
            return {t: 1 / 11 for t in range(2, 13)}
        return {t: w / total_w for t, w in weights.items()}

    def reshuffle(self) -> None:
        """Force a manual reshuffle (also triggered automatically)."""
        self._reshuffle()

    # ------------------------------------------------------------------
    # Read-only properties (useful for RL observation / UI)
    # ------------------------------------------------------------------

    @property
    def cards_left(self) -> int:
        return self._cards_left

    @property
    def recent_rolls(self) -> list[int]:
        return list(self._recent_rolls)

    def sevens_by_player(self) -> dict[int, int]:
        return dict(self._total_sevens)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reshuffle(self) -> None:
        self._deck       = {t: list(pairs) for t, pairs in _STANDARD_DECK.items()}
        self._cards_left = 36

    def _compute_weights(self, player_id: int) -> dict[int, float]:
        base = {t: len(self._deck[t]) / self._cards_left for t in range(2, 13)}
        return self._apply_adjustments(base, player_id)

    def _apply_adjustments(
        self, base: dict[int, float], player_id: int
    ) -> dict[int, float]:
        w = dict(base)

        # 1. Recent-roll penalty
        for t in range(2, 13):
            reduction = self._recent_count[t] * PROB_REDUCTION_RECENT
            w[t] = max(0.0, w[t] * (1.0 - reduction))

        # 2. Seven fairness (1v1 or N-player)
        if self._num_players >= 2:
            adj = self._seven_adjustment(player_id)
            adj = max(MIN_SEVEN_ADJUSTMENT, min(MAX_SEVEN_ADJUSTMENT, adj))
            w[7] *= adj

        return w

    def _seven_adjustment(self, player_id: int) -> float:
        return self._imbalance_adjustment(player_id) + self._streak_adjustment(player_id)

    def _imbalance_adjustment(self, player_id: int) -> float:
        total_sevens = sum(self._total_sevens.values())
        if total_sevens < self._num_players:
            return 1.0
        pct       = self._total_sevens[player_id] / total_sevens
        ideal_pct = 1.0 / self._num_players
        return 1.0 + (ideal_pct - pct) / ideal_pct

    def _streak_adjustment(self, player_id: int) -> float:
        if self._streak_count == 0:
            return 0.0
        direction = -1 if self._streak_player == player_id else 1
        return PROB_REDUCTION_STREAK * self._streak_count * direction

    def _weighted_choice(self, weights: dict[int, float]) -> int:
        totals  = list(range(2, 13))
        ws      = [weights[t] for t in totals]
        total_w = sum(ws)

        if total_w <= 0:
            non_empty = [t for t in totals if self._deck[t]]
            return self._rng.choice(non_empty)

        r = self._rng.random() * total_w
        for t, w in zip(totals, ws):
            if r <= w:
                return t
            r -= w
        return totals[-1]

    def _update_recent(self, total: int) -> None:
        self._recent_rolls.append(total)
        self._recent_count[total] += 1
        if len(self._recent_rolls) > MAX_RECENT_MEMORY:
            evicted = self._recent_rolls.pop(0)
            self._recent_count[evicted] -= 1

    def _update_sevens(self, player_id: int) -> None:
        self._total_sevens[player_id] += 1
        if self._streak_player == player_id:
            self._streak_count += 1
        else:
            self._streak_player = player_id
            self._streak_count  = 1
