"""
watch_game.py

Two outputs:
  1. action_freq.png  — bar chart of action types the bot uses vs opponent
  2. Console game log — verbose play-by-play of one full game

Usage:
    # PPO bot vs heuristic opponent
    python watch_game.py checkpoints/ckpt_final.pt
    python watch_game.py checkpoints/ckpt_final.pt --opponent ows --games 50

    # MCTS bot vs heuristic opponent (no checkpoint needed)
    python watch_game.py --bot mcts --opponent ows --games 5
    python watch_game.py --bot mcts --opponent ows --mcts-sims 100 --games 10

    # PPO bot vs MCTS opponent (slow — use few games)
    python watch_game.py checkpoints/ckpt_final.pt --opponent mcts --games 3

Bots:      ppo (default) | mcts
Opponents: random | road_builder | ows | balanced | mcts
"""

from __future__ import annotations
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
for _p in [str(_root), str(_root / "training")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import collections
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from env.actions import (
    action_mask, decode_action,
    SETTLE_START, CITY_START, ROAD_START, ROLL, ROBBER_START,
    BUY_DEV, PLAY_KNIGHT, PLAY_RB, YOP_START, MONO_START,
    TRADE_START, END_TURN, DISCARD_START, ACTION_DIM,
)
from env.catan_env import CatanEnv, encode_observation
from env.game_state import visible_vp, total_vp
from env.policy_net import PolicyNet
from heuristic_players import heuristic_action, STRATEGIES
from mcts_bot import mcts_action

_OPP_CHOICES = ["random", "mcts"] + list(STRATEGIES)
_BOT_CHOICES = ["ppo", "mcts"]


# ---------------------------------------------------------------------------
# Action type label
# ---------------------------------------------------------------------------

def action_type(action_id: int) -> str:
    name, _ = decode_action(action_id)
    return name


# ---------------------------------------------------------------------------
# Play one game, return action log and per-turn VP history
# ---------------------------------------------------------------------------

def play_game(
    policy:     "PolicyNet | None",
    policy_pid: int,
    seed:       int,
    device:     torch.device,
    verbose:    bool = False,
    opponent:   str  = "random",
    bot:        str  = "ppo",
    mcts_sims:  int  = 200,
) -> dict:
    """
    Returns {
      "winner": int,
      "steps":  int,
      "policy_actions": List[str],    # action types taken by bot
      "opponent_actions": List[str],
      "policy_vp": List[int],
      "opponent_vp": List[int],
    }

    bot:      "ppo" | "mcts"
    opponent: "random" | "road_builder" | "ows" | "balanced" | "mcts"
    """
    rng = np.random.default_rng(seed)
    env = CatanEnv(seed=seed)
    env.reset()

    opp_pid   = 1 - policy_pid
    opp_label = opponent.upper() if opponent not in ("random", "mcts") else opponent.upper()

    policy_actions:   List[str] = []
    opponent_actions: List[str] = []
    policy_vp_hist:   List[int] = []
    opponent_vp_hist: List[int] = []

    for step in range(20_000):
        pid     = env.state.current_player
        obs_np  = encode_observation(env.state, pid)
        mask_np = action_mask(env.state)

        if pid == policy_pid:
            # --- Bot's turn ---
            if bot == "mcts":
                action = mcts_action(env.state, pid, rng, n_simulations=mcts_sims)
            else:
                # PPO policy
                if verbose:
                    with torch.no_grad():
                        obs_t  = torch.as_tensor(obs_np).unsqueeze(0).to(device)
                        mask_t = torch.as_tensor(mask_np).unsqueeze(0).to(device)
                        dist, value = policy(obs_t, mask_t)
                        probs  = dist.probs[0].cpu().numpy()
                    action = int(dist.sample().item())
                else:
                    action, _, _ = policy.act(obs_np, mask_np)

            atype = action_type(action)
            policy_actions.append(atype)
            policy_vp_hist.append(visible_vp(env.state, pid))

            if verbose:
                bot_label = "MCTS" if bot == "mcts" else "BOT"
                vp_self   = visible_vp(env.state, pid)
                vp_opp    = visible_vp(env.state, opp_pid)
                res       = env.state.players[pid].resources
                res_str   = " ".join(f"{r.name[0]}{v}" for r, v in res.items() if v > 0) or "none"
                print(
                    f"  [{bot_label} p{pid}] turn={env.state.turn_number:3d}  "
                    f"VP={vp_self}/{vp_opp}  res=[{res_str}]  "
                    f"phase={env.state.phase.name}"
                )
                if bot == "ppo":
                    top5_idx = np.argsort(probs)[::-1][:5]
                    top5     = [(action_type(i), f"{probs[i]:.3f}") for i in top5_idx]
                    print(f"    -> {atype}   top5: {top5}")
                else:
                    print(f"    -> {atype}")
        else:
            # --- Opponent's turn ---
            if opponent == "random":
                legal  = np.where(mask_np)[0]
                action = int(rng.choice(legal))
            elif opponent == "mcts":
                action = mcts_action(env.state, pid, rng, n_simulations=mcts_sims)
            else:
                action = heuristic_action(env.state, opponent, rng)

            atype = action_type(action)
            opponent_actions.append(atype)
            opponent_vp_hist.append(visible_vp(env.state, pid))

            if verbose:
                print(f"  [{opp_label} p{pid}] {atype}")

        _, _, done, _, info = env.step(action)
        if done:
            winner = info["winner"]
            if verbose:
                print(f"\n  === GAME OVER: winner=player {winner} ===\n")
            return {
                "winner":           winner,
                "steps":            step + 1,
                "policy_actions":   policy_actions,
                "opponent_actions": opponent_actions,
                "policy_vp":        policy_vp_hist,
                "opponent_vp":      opponent_vp_hist,
            }

    return {
        "winner": -1, "steps": 20_000,
        "policy_actions":   policy_actions,
        "opponent_actions": opponent_actions,
        "policy_vp":        policy_vp_hist,
        "opponent_vp":      opponent_vp_hist,
    }


# ---------------------------------------------------------------------------
# Frequency chart
# ---------------------------------------------------------------------------

ACTION_ORDER = [
    "place_settlement", "place_city", "place_road",
    "roll_dice", "end_turn",
    "move_robber", "discard_resource",
    "buy_dev_card", "play_knight", "play_road_building",
    "play_year_of_plenty", "play_monopoly", "bank_trade",
]

def make_freq_chart(
    policy_counts: collections.Counter,
    random_counts: collections.Counter,
    save_path: str,
) -> None:
    total_p = max(sum(policy_counts.values()), 1)
    total_r = max(sum(random_counts.values()), 1)

    labels  = ACTION_ORDER
    p_freq  = [policy_counts.get(l, 0) / total_p for l in labels]
    r_freq  = [random_counts.get(l, 0) / total_r for l in labels]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0d1117")
    ax.set_facecolor("#161b22")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    bars_p = ax.bar(x - width / 2, p_freq, width, label="Bot",    color="#e76f51", alpha=0.85)
    bars_r = ax.bar(x + width / 2, r_freq, width, label="Random", color="#4e9af1", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [l.replace("_", "\n") for l in labels],
        color="#c9d1d9", fontsize=8
    )
    ax.set_ylabel("Fraction of all actions", color="#8b949e")
    ax.set_title("Bot vs Random — Action Type Frequency", color="white", fontsize=13)
    ax.tick_params(colors="#8b949e")
    ax.yaxis.label.set_color("#8b949e")
    ax.tick_params(axis="y", colors="#8b949e")
    ax.legend(facecolor="#0d1117", edgecolor="#30363d", labelcolor="#c9d1d9")

    # Value labels
    for bar in bars_p:
        h = bar.get_height()
        if h > 0.005:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.002, f"{h:.2f}",
                    ha="center", va="bottom", color="#c9d1d9", fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved -> {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="Path to PPO checkpoint (required when --bot ppo)")
    p.add_argument("--bot",      type=str,  default="ppo",
                   choices=_BOT_CHOICES,
                   help="Bot type: ppo (default) | mcts")
    p.add_argument("--games",    type=int,  default=50)
    p.add_argument("--seed",     type=int,  default=0)
    p.add_argument("--verbose",  action="store_true",
                   help="Print full play-by-play for game 0")
    p.add_argument("--opponent", type=str,  default="random",
                   choices=_OPP_CHOICES,
                   help="Opponent: random | road_builder | ows | balanced | mcts")
    p.add_argument("--mcts-sims", type=int, default=200, dest="mcts_sims",
                   help="Simulations per MCTS move decision (default 200). "
                        "Use fewer (50-100) for faster multi-game runs.")
    cfg = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy only when needed
    policy = None
    if cfg.bot == "ppo":
        if cfg.checkpoint is None:
            p.error("checkpoint is required when --bot ppo")
        ckpt   = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
        policy = PolicyNet().to(device)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()

    if cfg.opponent == "mcts" or cfg.bot == "mcts":
        print(f"MCTS simulations per move: {cfg.mcts_sims}")
        if cfg.mcts_sims >= 200 and cfg.games > 5:
            print(f"  Warning: {cfg.games} games x MCTS will be slow. "
                  f"Consider --games 5 or --mcts-sims 50.\n")

    policy_counts:   collections.Counter = collections.Counter()
    opponent_counts: collections.Counter = collections.Counter()
    wins = 0

    bot_label = "MCTS" if cfg.bot == "mcts" else "PPO Bot"
    opp_label = cfg.opponent.replace("_", " ").title() if cfg.opponent not in ("random", "mcts") else cfg.opponent.upper()
    print(f"Playing {cfg.games} games: {bot_label} vs {opp_label} (alternating sides)...\n")

    for g in range(cfg.games):
        policy_pid = g % 2
        verbose    = cfg.verbose and g == 0
        if verbose:
            print(f"=== GAME 0 ({bot_label} plays as player {policy_pid}) ===")

        result = play_game(
            policy, policy_pid, seed=cfg.seed + g, device=device,
            verbose=verbose, opponent=cfg.opponent,
            bot=cfg.bot, mcts_sims=cfg.mcts_sims,
        )

        if result["winner"] == policy_pid:
            wins += 1

        for a in result["policy_actions"]:
            policy_counts[a] += 1
        for a in result["opponent_actions"]:
            opponent_counts[a] += 1

    win_rate = wins / cfg.games
    print(f"\nResults over {cfg.games} games  ({bot_label} vs {opp_label}):")
    print(f"  Win rate ({bot_label}): {win_rate:.3f}  ({wins}/{cfg.games})")
    print(f"\n{bot_label} action breakdown:")
    total = sum(policy_counts.values())
    for atype in ACTION_ORDER:
        n = policy_counts.get(atype, 0)
        if n:
            print(f"  {atype:<25s}  {n:6d}  ({100*n/total:5.1f}%)")

    make_freq_chart(policy_counts, opponent_counts, "action_freq.png")


if __name__ == "__main__":
    main()
