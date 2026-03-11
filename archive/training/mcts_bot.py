"""
mcts_bot.py

Flat Monte Carlo with UCB action selection for 1v1 Catan.

Strategy
--------
For each legal action in the current state, run heuristic rollouts and score
the resulting position with compute_phi().  UCB1 allocates more simulations to
actions that look promising, exactly like a one-level MCTS tree.

Stochasticity (dice) is handled naturally: each deep-copied simulation has its
own freshly seeded BalancedDiceEngine, so every rollout samples a different
dice sequence without explicit determinization.

Public API
----------
    from mcts_bot import mcts_action

    action = mcts_action(state, pid, rng)
    action = mcts_action(state, pid, rng, n_simulations=100, rollout_depth=40)

Parameters
----------
n_simulations : int
    Total rollout budget shared across all legal actions (default 200).
rollout_depth : int
    Maximum number of actions per rollout before phi evaluation (default 50).
rollout_strategy : str
    Heuristic used by BOTH sides during rollouts: "ows" | "balanced" (default "ows").
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy
import math
import random
from typing import Dict, List, Optional

import numpy as np

from env.actions import action_mask
from env.balanced_dice import BalancedDiceEngine
from env.catan_env import apply_action
from env.game_state import GameState, GamePhase
from env.potential_function import compute_phi
from env.reward_config import RewardConfig
from heuristic_players import heuristic_action

# Shared reward config for phi scoring (uses all RewardConfig defaults)
_PHI_CFG = RewardConfig()

# UCB1 exploration constant — slightly above sqrt(2) to encourage exploration
# across Catan's diverse action types
_UCB_C = 1.5


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _reseed(state: GameState, seed: int) -> None:
    """Replace the RNG engines in a copied state with fresh seeded ones."""
    state.rng  = random.Random(seed)
    state.dice = BalancedDiceEngine(num_players=2, seed=seed ^ 0xDEAD_BEEF)


def _terminal_score(state: GameState, pid: int) -> Optional[float]:
    """Return ±1e6 if the game is over, else None."""
    if state.phase == GamePhase.GAME_OVER:
        return 1e6 if state.winner == pid else -1e6
    return None


def _rollout(
    state:    GameState,
    pid:      int,
    depth:    int,
    strategy: str,
    sim_rng:  np.random.Generator,
) -> float:
    """
    Play up to `depth` heuristic actions from `state` (mutates in place).
    Both players use the same heuristic strategy.

    Returns the score from pid's perspective:
      +1e6  game ended in pid's win
      -1e6  game ended in pid's loss
      phi   position evaluation if rollout depth is reached
    """
    for _ in range(depth):
        t = _terminal_score(state, pid)
        if t is not None:
            return t
        if not action_mask(state).any():
            break
        a = heuristic_action(state, strategy, sim_rng)
        apply_action(state, int(a))

    t = _terminal_score(state, pid)
    return t if t is not None else compute_phi(state, pid, _PHI_CFG)


def _simulate(
    root_state: GameState,
    action:     int,
    pid:        int,
    depth:      int,
    strategy:   str,
    seed:       int,
) -> float:
    """
    Deep-copy root_state, apply action, run a rollout.
    Returns the score from pid's perspective.
    """
    sim = copy.deepcopy(root_state)
    _reseed(sim, seed)

    winner = apply_action(sim, action)
    if winner is not None:
        return 1e6 if winner == pid else -1e6

    sim_rng = np.random.default_rng(seed ^ 0xCAFE)
    return _rollout(sim, pid, depth, strategy, sim_rng)


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def mcts_action(
    state:            GameState,
    pid:              int,
    rng:              np.random.Generator,
    n_simulations:    int = 200,
    rollout_depth:    int = 50,
    rollout_strategy: str = "ows",
) -> int:
    """
    Return the best legal action for pid using flat Monte Carlo + UCB.

    Args:
        state:            Current game state (not modified).
        pid:              Player id making the decision.
        rng:              Numpy RNG used to generate simulation seeds.
        n_simulations:    Total simulation budget across all actions.
        rollout_depth:    Max heuristic steps per rollout before phi scoring.
        rollout_strategy: Heuristic strategy used for both sides in rollouts.

    Returns:
        Best legal action index (int).
    """
    mask  = action_mask(state)
    legal = list(np.where(mask)[0])

    if len(legal) == 1:
        return legal[0]

    # Clamp budget so we always have at least one sim per action
    budget = max(n_simulations, len(legal))

    # Per-action running totals and visit counts
    totals: Dict[int, float] = {a: 0.0 for a in legal}
    counts: Dict[int, int]   = {a: 0   for a in legal}

    # Generate all seeds upfront for reproducibility
    seeds = rng.integers(0, 2**31, size=budget + 1)
    sidx  = 0

    # --- Phase 1: one simulation per action (initialise UCB stats) ---
    for a in legal:
        totals[a] += _simulate(
            state, a, pid, rollout_depth, rollout_strategy, int(seeds[sidx])
        )
        counts[a]  = 1
        sidx += 1

    # --- Phase 2: UCB-guided remaining budget ---
    for _ in range(budget - len(legal)):
        total_n = sum(counts.values())
        log_n   = math.log(total_n)

        best_a = max(
            legal,
            key=lambda a: (
                totals[a] / counts[a]
                + _UCB_C * math.sqrt(log_n / counts[a])
            ),
        )

        totals[best_a] += _simulate(
            state, best_a, pid, rollout_depth, rollout_strategy, int(seeds[sidx])
        )
        counts[best_a] += 1
        sidx += 1

    # Return the action with the highest mean score
    return max(legal, key=lambda a: totals[a] / counts[a])
