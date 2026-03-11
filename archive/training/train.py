"""
train.py

Self-play PPO training for CatanBot.

Features:
  - Vectorised environments (n_envs parallel games, batched GPU inference)
  - Linear LR and entropy annealing
  - Evaluation vs random agent at configurable intervals
  - Resume from checkpoint

Usage:
    python train.py
    python train.py --n-envs 8 --total-steps 10_000_000
    python train.py --resume checkpoints/ckpt_500000.pt
"""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.catan_env import CatanEnv, encode_observation, OBS_DIM
from env.actions import action_mask, ACTION_DIM
from env.game_state import GamePhase
from env.policy_net import PolicyNet
from env.reward_config import RewardConfig
from env.reward_calculator import take_snapshot, compute_reward
from env.potential_function import compute_phi
from heuristic_players import heuristic_action as _ows_heuristic_action


# Setup phases — OWS heuristic handles these; PPO only trains on main game
_SETUP_PHASES = frozenset({
    GamePhase.SETUP_P0_SETTLE_1, GamePhase.SETUP_P1_SETTLE_1,
    GamePhase.SETUP_P0_SETTLE_2, GamePhase.SETUP_P1_SETTLE_2,
    GamePhase.SETUP_P0_ROAD_1,   GamePhase.SETUP_P1_ROAD_1,
    GamePhase.SETUP_P0_ROAD_2,   GamePhase.SETUP_P1_ROAD_2,
})


def _flush_setup(env: CatanEnv, rng: np.random.Generator) -> None:
    """Run all pending setup actions via OWS heuristic. Never writes to the PPO buffer."""
    while env.state.phase in _SETUP_PHASES:
        action = _ows_heuristic_action(env.state, "ows", rng)
        env.step(action)


# ---------------------------------------------------------------------------
# Hyperparameters (all overridable via CLI)
# ---------------------------------------------------------------------------

DEFAULTS = dict(
    total_steps    = 5_000_000,
    n_envs         = 8,          # parallel environments
    rollout_steps  = 2048,       # steps per env before each PPO update
    ppo_epochs     = 4,
    batch_size     = 512,
    lr             = 3e-4,
    lr_end         = 3e-5,       # linearly annealed to this
    gamma          = 0.99,
    gae_lambda     = 0.95,
    clip_eps       = 0.2,
    value_coef     = 0.1,
    entropy_coef   = 0.10,       # starting entropy coef (annealed)
    entropy_end    = 0.010,      # final entropy coef
    max_grad_norm    = 0.5,
    # Reward config overrides (see env/reward_config.py for full list)
    win_reward       = 100.0,    # terminal win reward (spec default)
    potential_lambda = 0.05,     # potential shaping coefficient
    max_ep_steps     = 5000,     # hard-reset episode if it exceeds this
    save_every     = 500_000,    # checkpoint interval in env steps
    log_every      = 10,         # print every N PPO updates
    eval_freq      = 500_000,    # eval vs random interval; 0 = disabled
    eval_games     = 100,
    seed           = 42,
    save_dir       = str(Path(__file__).parent / "checkpoints"),
    resume         = None,       # path to checkpoint to resume from
    setup_ppo      = False,      # Stage 2: let PPO also learn setup phase
    heuristic_frac = 0.5,        # fraction of envs playing vs OWS heuristic opponent
    heuristic_opp  = "ows",      # which heuristic strategy to use as opponent
)


# ---------------------------------------------------------------------------
# Vectorised environment
# ---------------------------------------------------------------------------

class VecEnv:
    """N CatanEnv instances stepped synchronously."""

    def __init__(self, n: int, base_seed: int):
        self.envs = [CatanEnv(seed=base_seed + i) for i in range(n)]
        self.n    = n

    def reset_all(self) -> None:
        for env in self.envs:
            env.reset()

    def reset_one(self, i: int) -> None:
        self.envs[i].reset()


# ---------------------------------------------------------------------------
# GAE advantage computation
# ---------------------------------------------------------------------------

def compute_gae(
    values:     np.ndarray,
    rewards:    np.ndarray,
    dones:      np.ndarray,
    last_value: float,
    gamma:      float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation.
    Returns (advantages, returns), both float32 arrays of length T.
    """
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    values_ext = np.append(values, last_value)

    last_gae = 0.0
    for t in reversed(range(T)):
        not_done  = 1.0 - float(dones[t])
        delta     = rewards[t] + gamma * values_ext[t + 1] * not_done - values[t]
        last_gae  = delta + gamma * gae_lambda * not_done * last_gae
        advantages[t] = last_gae

    return advantages, advantages + values


# ---------------------------------------------------------------------------
# PPO update
# ---------------------------------------------------------------------------

def ppo_update(
    policy:        PolicyNet,
    optimizer:     optim.Optimizer,
    obs:           np.ndarray,
    masks:         np.ndarray,
    actions:       np.ndarray,
    old_log_probs: np.ndarray,
    advantages:    np.ndarray,
    returns:       np.ndarray,
    clip_eps:      float,
    value_coef:    float,
    entropy_coef:  float,
    batch_size:    int,
    n_epochs:      int,
    max_grad_norm: float,
    device:        torch.device = torch.device("cpu"),
) -> dict:
    T     = len(obs)
    dev   = device
    obs_t    = torch.as_tensor(obs,           dtype=torch.float32).to(dev)
    masks_t  = torch.as_tensor(masks,         dtype=torch.bool).to(dev)
    acts_t   = torch.as_tensor(actions,       dtype=torch.long).to(dev)
    old_lp_t = torch.as_tensor(old_log_probs, dtype=torch.float32).to(dev)
    adv_t    = torch.as_tensor(advantages,    dtype=torch.float32).to(dev)
    ret_t    = torch.as_tensor(returns,       dtype=torch.float32).to(dev)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    total_pg = total_v = total_ent = 0.0
    n_updates = 0

    for _ in range(n_epochs):
        idx = torch.randperm(T)
        for start in range(0, T, batch_size):
            b = idx[start : start + batch_size]

            dist, value = policy(obs_t[b], masks_t[b])
            new_lp  = dist.log_prob(acts_t[b])
            entropy = dist.entropy().mean()

            # Clamp log-ratio before exp() to prevent overflow: if new_lp >> old_lp,
            # exp() hits inf, and inf * negative_advantage = -inf which PPO clip
            # cannot recover from, producing NaN gradients.
            log_ratio = (new_lp - old_lp_t[b]).clamp(-20.0, 20.0)
            ratio     = torch.exp(log_ratio)
            adv_b     = adv_t[b]
            pg_loss   = -torch.min(
                ratio * adv_b,
                torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv_b,
            ).mean()

            v_loss = 0.5 * (value - ret_t[b]).pow(2).mean()
            loss   = pg_loss + value_coef * v_loss - entropy_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()

            total_pg  += pg_loss.item()
            total_v   += v_loss.item()
            total_ent += entropy.item()
            n_updates += 1

    return {
        "pg_loss": total_pg  / n_updates,
        "v_loss":  total_v   / n_updates,
        "entropy": total_ent / n_updates,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: argparse.Namespace) -> None:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    os.makedirs(cfg.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device: {device}"
        + (f"  ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "")
    )

    policy    = PolicyNet().to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)

    total_steps  = 0
    next_save    = cfg.save_every
    next_eval    = cfg.eval_freq if cfg.eval_freq > 0 else float("inf")
    best_win_rate = -1.0

    # --- Resume from checkpoint ---
    if cfg.resume:
        ckpt = torch.load(cfg.resume, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["policy"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        total_steps = ckpt.get("steps", 0)
        next_save   = ((total_steps // cfg.save_every) + 1) * cfg.save_every
        if cfg.eval_freq > 0:
            next_eval = ((total_steps // cfg.eval_freq) + 1) * cfg.eval_freq
        print(f"Resumed from {cfg.resume} at step {total_steps}")

    T   = cfg.rollout_steps
    N   = cfg.n_envs
    vec = VecEnv(N, base_seed=cfg.seed)
    vec.reset_all()

    # OWS heuristic rng (separate stream so it doesn't affect PPO reproducibility)
    ows_rng = np.random.default_rng(cfg.seed + 7919)

    # Advance all envs past setup so the PPO buffer never sees setup states
    if not cfg.setup_ppo:
        for env in vec.envs:
            _flush_setup(env, ows_rng)

    # --- Opponent mixing (Option B) ---
    # First n_heuristic envs play PPO vs OWS heuristic; rest are self-play.
    n_heuristic = round(N * cfg.heuristic_frac)
    is_heuristic = [i < n_heuristic for i in range(N)]
    # ppo_pid[i]: which player_id PPO controls in env i.
    # For self-play envs this is unused (PPO acts for whoever is current_player).
    ppo_pid = [0] * N

    def _advance_heuristic_to_ppo_turn(i: int) -> None:
        """Step the heuristic opponent until it's PPO's turn (or game over)."""
        env = vec.envs[i]
        heuristic_pid = 1 - ppo_pid[i]
        while (env.state.phase != GamePhase.GAME_OVER
               and env.state.current_player == heuristic_pid):
            a = _ows_heuristic_action(env.state, cfg.heuristic_opp, ows_rng)
            env.step(a)

    def _init_heuristic_env(i: int) -> None:
        """Assign PPO a random side and advance past opening heuristic turns."""
        ppo_pid[i] = int(ows_rng.integers(0, 2))
        _advance_heuristic_to_ppo_turn(i)

    for i in range(N):
        if is_heuristic[i]:
            _init_heuristic_env(i)

    # Pre-allocated rollout buffers shaped (T, N, ...)
    buf_obs       = np.zeros((T, N, OBS_DIM),    dtype=np.float32)
    buf_masks     = np.zeros((T, N, ACTION_DIM), dtype=bool)
    buf_actions   = np.zeros((T, N),             dtype=np.int64)
    buf_log_probs = np.zeros((T, N),             dtype=np.float32)
    buf_values    = np.zeros((T, N),             dtype=np.float32)
    buf_rewards   = np.zeros((T, N),             dtype=np.float32)
    buf_dones     = np.zeros((T, N),             dtype=bool)

    # Build reward config from CLI overrides
    reward_cfg = RewardConfig(
        win_reward=cfg.win_reward,
        loss_reward=-cfg.win_reward,
        potential_lambda=cfg.potential_lambda,
        vp_gain_visible=1.5,
        vp_loss_visible=-1.5,
        vp_gain_hidden=1.5,
        longest_road_gain=3.0,
        longest_road_loss=-3.0,
        largest_army_gain=3.0,
        largest_army_loss=-3.0,
    )

    update_num     = 0
    n_episodes     = 0
    # Self-play: both players are PPO — track P0 wins as a balance sanity check
    n_selfplay_eps    = 0
    n_selfplay_p0wins = 0
    # Heuristic: PPO vs fixed OWS opponent
    n_heuristic_eps   = 0
    n_heuristic_wins  = 0
    ep_steps       = [0] * N
    ep_ret         = [0.0] * N
    recent_returns: List[float] = []
    t0             = time.time()

    def _end_episode(ei: int, w: Optional[int], timed_out: bool = False) -> None:
        nonlocal n_episodes, n_selfplay_eps, n_selfplay_p0wins
        nonlocal n_heuristic_eps, n_heuristic_wins
        if not timed_out:
            if is_heuristic[ei]:
                n_heuristic_eps += 1
                if w == ppo_pid[ei]:
                    n_heuristic_wins += 1
            else:
                n_selfplay_eps += 1
                if w == 0:
                    n_selfplay_p0wins += 1
        n_episodes += 1
        recent_returns.append(ep_ret[ei])
        if len(recent_returns) > 200:
            recent_returns.pop(0)
        ep_ret[ei]   = 0.0
        ep_steps[ei] = 0
        vec.reset_one(ei)
        if not cfg.setup_ppo:
            _flush_setup(vec.envs[ei], ows_rng)
        if is_heuristic[ei]:
            _init_heuristic_env(ei)

    while total_steps < cfg.total_steps:

        # --- Linear annealing ---
        frac    = max(0.0, 1.0 - total_steps / cfg.total_steps)
        lr_now  = cfg.lr_end  + (cfg.lr             - cfg.lr_end)  * frac
        ent_now = cfg.entropy_end + (cfg.entropy_coef - cfg.entropy_end) * frac
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        # --- Collect rollout (batched GPU inference per timestep) ---
        policy.eval()
        for t in range(T):
            obs_batch  = np.empty((N, OBS_DIM),    dtype=np.float32)
            mask_batch = np.empty((N, ACTION_DIM),  dtype=bool)
            pids       = []
            snapshots  = []

            for i, env in enumerate(vec.envs):
                pid = env.state.current_player
                pids.append(pid)
                phi  = compute_phi(env.state, pid, reward_cfg)
                snap = take_snapshot(env.state, pid, phi)
                snapshots.append(snap)
                obs_batch[i]  = encode_observation(env.state, pid)
                mask_batch[i] = action_mask(env.state)

            with torch.no_grad():
                obs_t  = torch.as_tensor(obs_batch).to(device)
                mask_t = torch.as_tensor(mask_batch).to(device)
                dist, values_t  = policy(obs_t, mask_t)
                actions_t   = dist.sample()
                log_probs_t = dist.log_prob(actions_t)

            actions_np   = actions_t.cpu().numpy()
            log_probs_np = log_probs_t.cpu().numpy()
            values_np    = values_t.cpu().numpy()

            for i, env in enumerate(vec.envs):
                _, _, done, _, info = env.step(int(actions_np[i]))

                pid    = pids[i]
                winner = info.get("winner")
                shaped, _ = compute_reward(
                    snapshots[i], int(actions_np[i]),
                    env.state, done, winner, reward_cfg,
                )

                ep_ret[i] += shaped

                buf_obs[t, i]       = obs_batch[i]
                buf_masks[t, i]     = mask_batch[i]
                buf_actions[t, i]   = actions_np[i]
                buf_log_probs[t, i] = log_probs_np[i]
                buf_values[t, i]    = values_np[i]
                buf_rewards[t, i]   = shaped
                buf_dones[t, i]     = done

                ep_steps[i]  += 1
                total_steps  += 1

                if done or ep_steps[i] >= cfg.max_ep_steps:
                    _end_episode(i, winner, timed_out=(ep_steps[i] >= cfg.max_ep_steps and not done))

                elif is_heuristic[i]:
                    # Run heuristic opponent's response turns until PPO is up again.
                    # If the heuristic wins during its turns, attach the terminal
                    # reward to the current PPO buffer slot and end the episode.
                    h_pid = 1 - ppo_pid[i]
                    env_h = vec.envs[i]
                    while (env_h.state.phase != GamePhase.GAME_OVER
                           and env_h.state.current_player == h_pid):
                        a = _ows_heuristic_action(env_h.state, cfg.heuristic_opp, ows_rng)
                        _, _, done_h, _, info_h = env_h.step(a)
                        if done_h:
                            w_h = info_h.get("winner")
                            # Attach terminal outcome to the PPO step just taken
                            term_r = cfg.win_reward if w_h == ppo_pid[i] else -cfg.win_reward
                            buf_rewards[t, i] += term_r
                            ep_ret[i]         += term_r
                            buf_dones[t, i]    = True
                            _end_episode(i, w_h)
                            break

        # --- Bootstrap last value per env ---
        last_values = []
        for i, env in enumerate(vec.envs):
            if not buf_dones[-1, i]:
                pid       = env.state.current_player
                obs_last  = encode_observation(env.state, pid)
                mask_last = action_mask(env.state)
                with torch.no_grad():
                    _, lv = policy(
                        torch.as_tensor(obs_last).unsqueeze(0).to(device),
                        torch.as_tensor(mask_last).unsqueeze(0).to(device),
                    )
                last_values.append(lv.item())
            else:
                last_values.append(0.0)

        # --- GAE per env, then flatten ---
        all_adv, all_ret = [], []
        for i in range(N):
            adv_i, ret_i = compute_gae(
                buf_values[:, i], buf_rewards[:, i], buf_dones[:, i],
                last_values[i], cfg.gamma, cfg.gae_lambda,
            )
            all_adv.append(adv_i)
            all_ret.append(ret_i)

        advantages     = np.concatenate(all_adv)
        returns        = np.concatenate(all_ret)
        obs_flat       = buf_obs.reshape(-1, OBS_DIM)
        masks_flat     = buf_masks.reshape(-1, ACTION_DIM)
        actions_flat   = buf_actions.reshape(-1)
        log_probs_flat = buf_log_probs.reshape(-1)

        # --- PPO update ---
        policy.train()
        stats = ppo_update(
            policy, optimizer,
            obs_flat, masks_flat, actions_flat, log_probs_flat,
            advantages, returns,
            cfg.clip_eps, cfg.value_coef, ent_now,
            cfg.batch_size, cfg.ppo_epochs, cfg.max_grad_norm,
            device=device,
        )

        update_num += 1

        # --- Logging ---
        if update_num % cfg.log_every == 0:
            sp_wr  = n_selfplay_p0wins / max(1, n_selfplay_eps)
            h_wr   = n_heuristic_wins  / max(1, n_heuristic_eps)
            mean_ret = float(np.mean(recent_returns)) if recent_returns else 0.0
            elapsed  = time.time() - t0
            sps      = total_steps / max(elapsed, 1e-9)
            print(
                f"update={update_num:5d}  steps={total_steps:8d}  "
                f"sps={sps:6.0f}  eps={n_episodes:5d}  "
                f"sp_win={sp_wr:.3f}  h_win={h_wr:.3f}  ret={mean_ret:+.3f}  "
                f"pg={stats['pg_loss']:+.4f}  v={stats['v_loss']:.4f}  "
                f"ent={stats['entropy']:.4f}  lr={lr_now:.2e}"
            )

        # --- Checkpoint ---
        if total_steps >= next_save:
            path = os.path.join(cfg.save_dir, f"ckpt_{total_steps}.pt")
            torch.save({
                "steps":     total_steps,
                "policy":    policy.state_dict(),
                "optimizer": optimizer.state_dict(),
            }, path)
            print(f"  -> saved {path}")
            next_save += cfg.save_every

        # --- Eval vs random ---
        if total_steps >= next_eval:
            from evaluate import evaluate_vs_random
            policy.eval()
            res = evaluate_vs_random(
                policy, cfg.eval_games, device, seed=cfg.seed + update_num
            )
            wr = res['win_rate']
            best_tag = ""
            if wr > best_win_rate:
                best_win_rate = wr
                best_path = os.path.join(cfg.save_dir, "ckpt_best.pt")
                torch.save({
                    "steps":        total_steps,
                    "policy":       policy.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "win_vs_random": wr,
                }, best_path)
                best_tag = f"  ** NEW BEST -> {best_path}"
            print(
                f"  EVAL  win_vs_random={wr:.3f}  "
                f"avg_steps={res['avg_steps']:.0f}  games={res['n_games']}{best_tag}"
            )
            policy.train()
            next_eval += cfg.eval_freq

    # Final checkpoint
    path = os.path.join(cfg.save_dir, "ckpt_final.pt")
    torch.save({"steps": total_steps, "policy": policy.state_dict()}, path)
    print(f"Training complete. Final checkpoint: {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CatanBot self-play PPO trainer")
    p.add_argument("--total-steps",   type=int,   default=DEFAULTS["total_steps"])
    p.add_argument("--n-envs",        type=int,   default=DEFAULTS["n_envs"],
                   dest="n_envs")
    p.add_argument("--rollout",       type=int,   default=DEFAULTS["rollout_steps"],
                   dest="rollout_steps")
    p.add_argument("--epochs",        type=int,   default=DEFAULTS["ppo_epochs"],
                   dest="ppo_epochs")
    p.add_argument("--batch-size",    type=int,   default=DEFAULTS["batch_size"])
    p.add_argument("--lr",            type=float, default=DEFAULTS["lr"])
    p.add_argument("--lr-end",        type=float, default=DEFAULTS["lr_end"])
    p.add_argument("--gamma",         type=float, default=DEFAULTS["gamma"])
    p.add_argument("--gae-lambda",    type=float, default=DEFAULTS["gae_lambda"])
    p.add_argument("--clip-eps",      type=float, default=DEFAULTS["clip_eps"])
    p.add_argument("--value-coef",    type=float, default=DEFAULTS["value_coef"])
    p.add_argument("--entropy-coef",  type=float, default=DEFAULTS["entropy_coef"])
    p.add_argument("--entropy-end",   type=float, default=DEFAULTS["entropy_end"])
    p.add_argument("--max-grad-norm",    type=float, default=DEFAULTS["max_grad_norm"])
    p.add_argument("--win-reward",       type=float, default=DEFAULTS["win_reward"],
                   help="Terminal win reward (loss = -win_reward)")
    p.add_argument("--potential-lambda", type=float, default=DEFAULTS["potential_lambda"],
                   dest="potential_lambda",
                   help="Coefficient for potential-based shaping")
    p.add_argument("--max-ep-steps",     type=int,   default=DEFAULTS["max_ep_steps"])
    p.add_argument("--save-every",    type=int,   default=DEFAULTS["save_every"])
    p.add_argument("--log-every",     type=int,   default=DEFAULTS["log_every"])
    p.add_argument("--eval-freq",     type=int,   default=DEFAULTS["eval_freq"])
    p.add_argument("--eval-games",    type=int,   default=DEFAULTS["eval_games"])
    p.add_argument("--seed",          type=int,   default=DEFAULTS["seed"])
    p.add_argument("--save-dir",      type=str,   default=DEFAULTS["save_dir"])
    p.add_argument("--resume",        type=str,   default=DEFAULTS["resume"])
    p.add_argument("--setup-ppo",     action="store_true", default=DEFAULTS["setup_ppo"],
                   dest="setup_ppo",
                   help="Stage 2: let PPO also learn setup phase (default: OWS heuristic handles setup)")
    p.add_argument("--heuristic-frac", type=float, default=DEFAULTS["heuristic_frac"],
                   dest="heuristic_frac",
                   help="Fraction of envs playing vs OWS heuristic (0=pure self-play, 1=all vs heuristic)")
    p.add_argument("--heuristic-opp",  type=str,   default=DEFAULTS["heuristic_opp"],
                   dest="heuristic_opp",
                   help="Heuristic strategy to use as opponent (ows|balanced|road_builder)")
    return p.parse_args()


if __name__ == "__main__":
    train(_parse())
