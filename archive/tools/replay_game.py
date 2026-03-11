"""
replay_game.py

Interactive step-through board visualizer for a bot vs opponent game.

Usage:
    python replay_game.py checkpoints/ckpt_best.pt
    python replay_game.py checkpoints/ckpt_best.pt --seed 7
    python replay_game.py checkpoints/ckpt_best.pt --seed 7 --bot-pid 1
    python replay_game.py checkpoints/ckpt_best.pt --opponent road_builder
    python replay_game.py checkpoints/ckpt_best.pt --opponent ows --seed 42
    python replay_game.py checkpoints/ckpt_best.pt --opponent balanced

Opponents: random | road_builder | ows | balanced

Controls:
    Right arrow / N  — next action
    Left  arrow / P  — previous action
    Home             — jump to start
    End              — jump to end
    Click Next/Prev buttons with mouse

The window shows:
    Left panel  — Catan board with settlements, cities, roads, robber
    Right panel — Game log (action taken, resources, VP, turn info)
"""

from __future__ import annotations
import sys
from pathlib import Path
_root = Path(__file__).resolve().parent.parent
for _p in [str(_root), str(_root / "training"), str(_root / "tools")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import argparse
import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("TkAgg")          # interactive backend; falls back gracefully
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon, FancyArrowPatch
from matplotlib.widgets import Button
import numpy as np
import torch

from env.actions import action_mask, decode_action
from env.board import PORT_SLOT_DEFINITIONS, _EDGE_LIST as _BOARD_EDGE_LIST
from env.catan_env import CatanEnv, encode_observation
from env.game_state import visible_vp, Resource, DevCard, GamePhase
from env.policy_net import PolicyNet
from visualize_board import RESOURCE_COLORS, PIP_COUNT, PORT_COLORS, PORT_LABELS
from heuristic_players import heuristic_action, STRATEGIES
from mcts_bot import mcts_action

# ---------------------------------------------------------------------------
# Player colours
# ---------------------------------------------------------------------------

P_COLORS    = ["#e76f51", "#4e9af1"]   # orange / blue
P_NAMES     = ["Bot" if True else "?", "Random"]   # patched below
P_DARK      = ["#b34b2a", "#2a6fb3"]   # darker shade for cities


# ---------------------------------------------------------------------------
# Frame: snapshot of display data for one game state
# ---------------------------------------------------------------------------

@dataclass
class Frame:
    step:        int
    action_str:  str
    actor:       int          # player who just acted (0 or 1)
    phase:       str
    turn:        int
    roll:        Optional[int]
    settlements: List[List[int]] = field(default_factory=lambda: [[], []])
    cities:      List[List[int]] = field(default_factory=lambda: [[], []])
    roads:       List[List[int]] = field(default_factory=lambda: [[], []])
    robber_hex:  int = 0
    vp:          List[int] = field(default_factory=lambda: [0, 0])
    resources:   List[dict] = field(default_factory=lambda: [{}, {}])
    dev_hand:    List[List] = field(default_factory=lambda: [[], []])
    winner:      Optional[int] = None


def snapshot(env: CatanEnv, action_str: str, actor: int) -> Frame:
    s = env.state
    return Frame(
        step       = 0,            # filled in later
        action_str = action_str,
        actor      = actor,
        phase      = s.phase.name,
        turn       = s.turn_number,
        roll       = s.last_roll,
        settlements= [list(s.players[0].settlements), list(s.players[1].settlements)],
        cities     = [list(s.players[0].cities),      list(s.players[1].cities)],
        roads      = [list(s.players[0].roads),       list(s.players[1].roads)],
        robber_hex = s.robber_hex,
        vp         = [visible_vp(s, 0), visible_vp(s, 1)],
        resources  = [dict(s.players[0].resources),   dict(s.players[1].resources)],
        dev_hand   = [list(s.players[0].dev_hand),    list(s.players[1].dev_hand)],
        winner     = s.winner,
    )


def _action_label(action_id: int, env: CatanEnv) -> str:
    """Human-readable description of an action."""
    atype, param = decode_action(action_id)
    s = env.state
    pid = s.current_player

    if atype == "place_settlement":
        return f"place_settlement @ v{param}"
    if atype == "place_city":
        return f"place_city @ v{param}"
    if atype == "place_road":
        return f"place_road @ e{param}"
    if atype == "roll_dice":
        return "roll_dice"
    if atype == "move_robber":
        return f"move_robber → hex {param}"
    if atype == "discard_resource":
        return f"discard {param.name}"
    if atype == "buy_dev_card":
        return "buy_dev_card"
    if atype == "play_knight":
        return "play_knight"
    if atype == "play_road_building":
        return "play_road_building"
    if atype == "play_year_of_plenty":
        r1, r2 = param
        return f"year_of_plenty ({r1.name}+{r2.name})"
    if atype == "play_monopoly":
        return f"monopoly ({param.name})"
    if atype == "bank_trade":
        give, recv = param
        rate = 4   # approximate; exact rate not shown
        return f"trade {give.name} → {recv.name}"
    if atype == "end_turn":
        return "end_turn"
    return atype


# ---------------------------------------------------------------------------
# Play a game and record frames
# ---------------------------------------------------------------------------

def record_game(
    policy,
    policy_pid: int,
    seed: int,
    device: torch.device,
    opponent: str = "random",
    bot: str = "ppo",
    mcts_sims: int = 200,
) -> tuple[list[Frame], object]:   # frames, board
    rng = np.random.default_rng(seed)
    env = CatanEnv(seed=seed)
    env.reset()
    board = env.state.board

    # Initial frame (before any action)
    frames: list[Frame] = [snapshot(env, "(game start)", -1)]
    frames[0].step = 0

    for step in range(20_000):
        pid     = env.state.current_player
        mask_np = action_mask(env.state)

        if pid == policy_pid:
            if bot == "mcts":
                action = mcts_action(env.state, pid, rng, n_simulations=mcts_sims)
            else:
                obs_np = encode_observation(env.state, pid)
                action, _, _ = policy.act(obs_np, mask_np)
        else:
            if opponent == "random":
                legal  = np.where(mask_np)[0]
                action = int(rng.choice(legal))
            elif opponent == "mcts":
                action = mcts_action(env.state, pid, rng, n_simulations=mcts_sims)
            else:
                action = heuristic_action(env.state, opponent, rng)

        label = _action_label(action, env)
        actor = pid

        _, _, done, _, info = env.step(action)

        f = snapshot(env, label, actor)
        f.step = step + 1
        frames.append(f)

        if done:
            break

    return frames, board


# ---------------------------------------------------------------------------
# Board drawing helpers
# ---------------------------------------------------------------------------

SIZE = 1.0   # hex size


def _hex_center(h):
    cx = SIZE * (math.sqrt(3) * h.q + math.sqrt(3) / 2.0 * h.r)
    cy = SIZE * 1.5 * h.r
    return cx, cy


def _vertex_pos(board, vid):
    v = board.vertices[vid]
    return v.x, v.y


def draw_base_board(ax, board):
    """Draw static hex tiles, ports, and number tokens."""
    for h in board.hexes:
        cx, cy = _hex_center(h)
        verts = [
            (cx + SIZE * math.cos(math.radians(30 + 60 * i)),
             cy + SIZE * math.sin(math.radians(30 + 60 * i)))
            for i in range(6)
        ]
        poly = Polygon(verts, closed=True,
                       facecolor=RESOURCE_COLORS[h.resource],
                       edgecolor="#444444", linewidth=1.5, zorder=1)
        ax.add_patch(poly)

        if h.number_token > 0:
            num_color = "#c0392b" if h.number_token in (6, 8) else "#1a1a1a"
            circle = plt.Circle((cx, cy + 0.04), 0.28,
                                 color="#fffde7", ec="#aaaaaa", lw=1, zorder=2)
            ax.add_patch(circle)
            ax.text(cx, cy + 0.1, str(h.number_token),
                    fontsize=9, fontweight="bold", color=num_color,
                    ha="center", va="center", zorder=3)
            pips = PIP_COUNT.get(h.number_token, 0)
            for p in range(pips):
                px_ = cx + (p - (pips - 1) / 2.0) * 0.09
                ax.add_patch(plt.Circle((px_, cy - 0.1), 0.025,
                                         color=num_color, zorder=3))
        elif h.resource == "desert":
            ax.text(cx, cy, "R", fontsize=14, fontweight="bold",
                    color="#777777", ha="center", va="center", zorder=3)

    # Ports — draw before vertices so vertex dots render on top
    for eid in PORT_SLOT_DEFINITIONS:
        va_id, vb_id = _BOARD_EDGE_LIST[eid]
        av = board.vertices[va_id]
        bv = board.vertices[vb_id]
        ptype = av.port
        if ptype is None:
            continue
        color = PORT_COLORS.get(ptype, "#f5a623")
        label = PORT_LABELS.get(ptype, ptype)
        mx = (av.x + bv.x) / 2
        my = (av.y + bv.y) / 2
        dist = math.sqrt(mx * mx + my * my)
        nx, ny = (mx / dist, my / dist) if dist > 0 else (0, 1)
        lx, ly = mx + nx * 0.55, my + ny * 0.55
        ax.plot([lx, av.x], [ly, av.y], color=color, linewidth=2.5,
                solid_capstyle="round", zorder=3)
        ax.plot([lx, bv.x], [ly, bv.y], color=color, linewidth=2.5,
                solid_capstyle="round", zorder=3)
        ax.text(lx, ly, label, fontsize=8, fontweight="bold", color="#ffffff",
                ha="center", va="center", zorder=6, multialignment="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color,
                          edgecolor="white", linewidth=1.0, alpha=0.95))

    # Vertex dots — colored by port type
    for v in board.vertices:
        if v.port:
            color = PORT_COLORS.get(v.port, "#555555")
            ax.plot(v.x, v.y, "o", markersize=5.5, color=color,
                    markeredgecolor="#ffffff", markeredgewidth=0.8, zorder=4)
        else:
            ax.plot(v.x, v.y, "o", markersize=3, color="#555555",
                    markeredgecolor="#ffffff", markeredgewidth=0.5, zorder=4)


def draw_game_state(ax, board, frame: Frame):
    """Overlay settlements, cities, roads, and robber on the board axes."""
    # Roads
    for pid in range(2):
        for eid in frame.roads[pid]:
            va_id, vb_id = _BOARD_EDGE_LIST[eid]
            x0, y0 = _vertex_pos(board, va_id)
            x1, y1 = _vertex_pos(board, vb_id)
            ax.plot([x0, x1], [y0, y1],
                    color=P_COLORS[pid], linewidth=4.5,
                    solid_capstyle="round", zorder=5)

    # Settlements
    for pid in range(2):
        for vid in frame.settlements[pid]:
            x, y = _vertex_pos(board, vid)
            ax.plot(x, y, "^" if pid == 0 else "v",
                    markersize=13, color=P_COLORS[pid],
                    markeredgecolor="#ffffff", markeredgewidth=1.2, zorder=6)

    # Cities
    for pid in range(2):
        for vid in frame.cities[pid]:
            x, y = _vertex_pos(board, vid)
            ax.plot(x, y, "s",
                    markersize=14, color=P_DARK[pid],
                    markeredgecolor="#ffffff", markeredgewidth=1.5, zorder=7)

    # Robber
    h = board.hexes[frame.robber_hex]
    cx, cy = _hex_center(h)
    verts = [
        (cx + SIZE * math.cos(math.radians(30 + 60 * i)),
         cy + SIZE * math.sin(math.radians(30 + 60 * i)))
        for i in range(6)
    ]
    robber_border = Polygon(verts, closed=True, fill=False,
                            edgecolor="#8B0000", linewidth=3.5, zorder=8)
    ax.add_patch(robber_border)
    ax.text(cx, cy - 0.45, "ROBBER", fontsize=6, color="#8B0000",
            ha="center", va="center", fontweight="bold", zorder=9)


# ---------------------------------------------------------------------------
# Info panel text
# ---------------------------------------------------------------------------

RES_ABBR = {
    Resource.WOOD:  "W",
    Resource.BRICK: "B",
    Resource.SHEEP: "Sh",
    Resource.WHEAT: "Wh",
    Resource.ORE:   "O",
}

DEV_ABBR = {
    DevCard.KNIGHT:         "Kn",
    DevCard.ROAD_BUILDING:  "RB",
    DevCard.YEAR_OF_PLENTY: "YP",
    DevCard.MONOPOLY:       "Mo",
    DevCard.VICTORY_POINT:  "VP",
}


def _res_str(res_dict):
    parts = []
    for r in Resource:
        n = res_dict.get(r, 0)
        if n:
            parts.append(f"{RES_ABBR[r]}:{n}")
    return "  ".join(parts) or "—"


def _dev_str(hand):
    from collections import Counter
    c = Counter(hand)
    return "  ".join(f"{DEV_ABBR[k]}×{v}" for k, v in c.items()) or "—"


def build_info_text(frame: Frame, p_names: list[str], total: int) -> str:
    lines = []
    lines.append(f"Step {frame.step} / {total - 1}")
    lines.append(f"Turn {frame.turn}   Phase: {frame.phase}")
    if frame.roll is not None:
        lines.append(f"Last roll: {frame.roll}")
    lines.append("")

    actor_name = p_names[frame.actor] if frame.actor >= 0 else "—"
    lines.append(f"Action ({actor_name}):")
    lines.append(f"  {frame.action_str}")
    lines.append("")

    for pid in range(2):
        tag = " ← WINNER" if frame.winner == pid else ""
        lines.append(f"── {p_names[pid]} (P{pid}){tag} ──")
        lines.append(f"  VP: {frame.vp[pid]}")
        lines.append(f"  Resources: {_res_str(frame.resources[pid])}")
        lines.append(f"  Dev cards: {_dev_str(frame.dev_hand[pid])}")
        n_settle = len(frame.settlements[pid])
        n_city   = len(frame.cities[pid])
        n_road   = len(frame.roads[pid])
        lines.append(f"  Structs: {n_settle}×settle  {n_city}×city  {n_road}×road")
        lines.append("")

    if frame.winner is not None:
        lines.append(f"*** {p_names[frame.winner]} WINS ***")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive viewer
# ---------------------------------------------------------------------------

class Viewer:
    def __init__(self, frames: list[Frame], board, p_names: list[str]):
        self.frames  = frames
        self.board   = board
        self.p_names = p_names
        self.idx     = 0
        self.total   = len(frames)

        self.fig = plt.figure(figsize=(18, 10), facecolor="#0d1117")
        self.fig.canvas.manager.set_window_title("CatanBot Replay")

        # Layout: board on left (70%), info panel on right (30%)
        self.ax_board = self.fig.add_axes([0.01, 0.10, 0.63, 0.87])
        self.ax_info  = self.fig.add_axes([0.65, 0.10, 0.33, 0.87])
        self.ax_prev  = self.fig.add_axes([0.10, 0.02, 0.12, 0.055])
        self.ax_next  = self.fig.add_axes([0.24, 0.02, 0.12, 0.055])
        self.ax_home  = self.fig.add_axes([0.38, 0.02, 0.12, 0.055])
        self.ax_end   = self.fig.add_axes([0.52, 0.02, 0.12, 0.055])

        self.btn_prev = Button(self.ax_prev, "◀ Prev",  color="#21262d", hovercolor="#30363d")
        self.btn_next = Button(self.ax_next, "Next ▶",  color="#21262d", hovercolor="#30363d")
        self.btn_home = Button(self.ax_home, "⏮ Start", color="#21262d", hovercolor="#30363d")
        self.btn_end  = Button(self.ax_end,  "End ⏭",   color="#21262d", hovercolor="#30363d")

        for btn in (self.btn_prev, self.btn_next, self.btn_home, self.btn_end):
            btn.label.set_color("#c9d1d9")

        self.btn_prev.on_clicked(lambda _: self._go(self.idx - 1))
        self.btn_next.on_clicked(lambda _: self._go(self.idx + 1))
        self.btn_home.on_clicked(lambda _: self._go(0))
        self.btn_end.on_clicked(lambda _: self._go(self.total - 1))

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # Draw static base board once
        self._setup_board_axes()
        draw_base_board(self.ax_board, self.board)

        # Legend for players
        legend_patches = [
            mpatches.Patch(color=P_COLORS[0], label=f"P0: {p_names[0]} (▲ settle, ■ city)"),
            mpatches.Patch(color=P_COLORS[1], label=f"P1: {p_names[1]} (▼ settle, ■ city)"),
        ]
        self.ax_board.legend(handles=legend_patches, loc="lower left",
                             fontsize=8, framealpha=0.85,
                             facecolor="#161b22", edgecolor="#30363d",
                             labelcolor="#c9d1d9")

        self._record_base_counts()
        self._render()

    def _setup_board_axes(self):
        self.ax_board.set_facecolor("#0d1117")
        self.ax_board.set_aspect("equal")
        self.ax_board.axis("off")
        self.ax_info.set_facecolor("#161b22")
        self.ax_info.axis("off")

    def _on_key(self, event):
        if event.key in ("right", "n"):
            self._go(self.idx + 1)
        elif event.key in ("left", "p"):
            self._go(self.idx - 1)
        elif event.key == "home":
            self._go(0)
        elif event.key == "end":
            self._go(self.total - 1)

    def _go(self, new_idx: int):
        new_idx = max(0, min(new_idx, self.total - 1))
        if new_idx == self.idx:
            return
        self.idx = new_idx
        self._render()

    def _render(self):
        frame = self.frames[self.idx]

        # Clear game-state overlays (keep base board patches)
        # We'll remove only the overlays added by draw_game_state.
        # Simplest: collect artists beyond the base board draw count and remove them.
        # Remove overlay artists added after base board
        # Patches beyond base
        while len(self.ax_board.patches) > self._n_base_patches:
            self.ax_board.patches[-1].remove()
        while len(self.ax_board.lines) > self._n_base_lines:
            self.ax_board.lines[-1].remove()
        while len(self.ax_board.texts) > self._n_base_texts:
            self.ax_board.texts[-1].remove()

        draw_game_state(self.ax_board, self.board, frame)

        # Update title
        winner_tag = f"  —  {self.p_names[frame.winner]} WINS" if frame.winner is not None else ""
        self.ax_board.set_title(
            f"Step {self.idx}/{self.total-1}{winner_tag}",
            color="#c9d1d9", fontsize=11, pad=6,
        )

        # Info panel
        self.ax_info.cla()
        self.ax_info.set_facecolor("#161b22")
        self.ax_info.axis("off")
        info = build_info_text(frame, self.p_names, self.total)
        self.ax_info.text(
            0.05, 0.97, info,
            transform=self.ax_info.transAxes,
            fontsize=9, color="#c9d1d9",
            verticalalignment="top",
            fontfamily="monospace",
            wrap=True,
        )

        self.fig.canvas.draw_idle()

    def _record_base_counts(self):
        """Call once after draw_base_board, before first overlay."""
        self._n_base_patches = len(self.ax_board.patches)
        self._n_base_lines   = len(self.ax_board.lines)
        self._n_base_texts   = len(self.ax_board.texts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Interactive Catan game replay")
    p.add_argument("checkpoint", nargs="?", default=None,
                   help="Path to .pt checkpoint (required when --bot ppo)")
    p.add_argument("--bot",      type=str, default="ppo", choices=["ppo", "mcts"],
                   help="Bot type: ppo (default) | mcts")
    p.add_argument("--seed",     type=int, default=0,  help="Game seed")
    p.add_argument("--bot-pid",  type=int, default=0,  help="Which player the bot controls (0 or 1)",
                   dest="bot_pid")
    p.add_argument("--opponent", type=str, default="random",
                   choices=["random", "mcts"] + list(STRATEGIES),
                   help="Opponent strategy: random | mcts | road_builder | ows | balanced")
    p.add_argument("--mcts-sims", type=int, default=200, dest="mcts_sims",
                   help="MCTS simulations per move (default 200)")
    cfg = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = None
    if cfg.bot == "ppo":
        if cfg.checkpoint is None:
            p.error("checkpoint is required when --bot ppo")
        ckpt   = torch.load(cfg.checkpoint, map_location=device, weights_only=False)
        policy = PolicyNet().to(device)
        policy.load_state_dict(ckpt["policy"])
        policy.eval()

    bot_label = "MCTS" if cfg.bot == "mcts" else "Bot"
    opp_label = cfg.opponent.replace("_", " ").title() if cfg.opponent not in ("random", "mcts") else cfg.opponent.upper()
    p_names = [bot_label, opp_label]
    if cfg.bot_pid == 1:
        p_names = [opp_label, bot_label]

    print(f"Recording game (seed={cfg.seed}, bot={bot_label} P{cfg.bot_pid}, opponent={opp_label})...")
    if cfg.bot == "mcts" or cfg.opponent == "mcts":
        print(f"  MCTS sims/move: {cfg.mcts_sims}  (this may take a minute...)")
    frames, board = record_game(
        policy, cfg.bot_pid, cfg.seed, device,
        opponent=cfg.opponent, bot=cfg.bot, mcts_sims=cfg.mcts_sims,
    )
    print(f"Recorded {len(frames)} frames.")

    if frames[-1].winner is not None:
        print(f"Winner: {p_names[frames[-1].winner]}")
    else:
        print("Game did not finish (hit step limit).")

    viewer = Viewer(frames, board, p_names)
    plt.show()


if __name__ == "__main__":
    main()
