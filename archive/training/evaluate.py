"""
evaluate.py

Evaluate a trained policy against a uniform-random agent.

The policy alternates sides (player 0 / player 1) across games to give
an unbiased win-rate estimate.

Usage:
    python evaluate.py checkpoints/ckpt_final.pt
    python evaluate.py checkpoints/ckpt_final.pt --games 200 --seed 0
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import numpy as np
import torch

from env.catan_env import CatanEnv, encode_observation
from env.actions import action_mask
from env.policy_net import PolicyNet


def evaluate_vs_random(
    policy:  PolicyNet,
    n_games: int,
    device:  torch.device,
    seed:    int = 0,
) -> dict:
    """
    Play n_games against a random agent, alternating which side the
    policy occupies.

    Returns:
        {
          "win_rate":  float,   # fraction of games the policy won
          "avg_steps": float,   # mean game length in env steps
          "n_games":   int,
        }
    """
    policy.eval()
    rng   = np.random.default_rng(seed)
    wins  = 0
    steps_per_game: list[int] = []

    for game_i in range(n_games):
        env        = CatanEnv(seed=seed + game_i)
        env.reset()
        policy_pid = game_i % 2   # alternate: 0, 1, 0, 1, …

        for step in range(10_000):
            pid     = env.state.current_player
            obs_np  = encode_observation(env.state, pid)
            mask_np = action_mask(env.state)

            if pid == policy_pid:
                action, _, _ = policy.act(obs_np, mask_np)
            else:
                legal  = np.where(mask_np)[0]
                action = int(rng.choice(legal))

            _, _, done, _, info = env.step(action)

            if done:
                if info["winner"] == policy_pid:
                    wins += 1
                steps_per_game.append(step + 1)
                break

    return {
        "win_rate":  wins / n_games,
        "avg_steps": float(np.mean(steps_per_game)) if steps_per_game else 0.0,
        "n_games":   n_games,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate a CatanBot checkpoint vs random")
    p.add_argument("checkpoint", type=str, help="Path to .pt checkpoint file")
    p.add_argument("--games",    type=int, default=200)
    p.add_argument("--seed",     type=int, default=0)
    cfg = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt   = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
    policy = PolicyNet().to(device)
    policy.load_state_dict(ckpt["policy"])

    result = evaluate_vs_random(policy, cfg.games, device, cfg.seed)
    print(
        f"Checkpoint: {cfg.checkpoint}\n"
        f"Games:      {result['n_games']}\n"
        f"Win rate:   {result['win_rate']:.3f}  ({result['win_rate'] * 100:.1f}%)\n"
        f"Avg steps:  {result['avg_steps']:.0f}"
    )
