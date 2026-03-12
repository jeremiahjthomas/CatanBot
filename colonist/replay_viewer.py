"""
replay_viewer.py

Streamlit app to step through a Colonist.io replay action by action,
visualising the board state (settlements, cities, roads, robber) at each step.

Usage:
    streamlit run colonist/replay_viewer.py
"""

import sys
import io
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPLAYS_DIR = Path(__file__).parent / "replays"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOURCE_NAMES = {0: "desert", 1: "wood", 2: "brick", 3: "sheep", 4: "wheat", 5: "ore"}
RESOURCE_COLORS = {
    "desert": "#d4b483",
    "wood":   "#4a7c3f",
    "brick":  "#c44b2b",
    "sheep":  "#8cc84b",
    "wheat":  "#f0c030",
    "ore":    "#8a8a8a",
}
PORT_NAMES = {1: "3:1", 2: "wood 2:1", 3: "brick 2:1", 4: "sheep 2:1", 5: "wheat 2:1", 6: "ore 2:1"}
DEV_CARD_NAMES = {0: "knight", 1: "road_building", 2: "year_of_plenty", 3: "monopoly", 4: "vp"}

# Colonist player colours → matplotlib colours
PLAYER_MCOLORS = {1: "#e53935", 5: "#1e88e5", 2: "#fb8c00", 3: "#ffffff", 4: "#43a047"}
PLAYER_EDGE_COLORS = {1: "#b71c1c", 5: "#0d47a1", 2: "#e65100", 3: "#9e9e9e", 4: "#1b5e20"}

SQRT3 = math.sqrt(3)

LOG_TYPE_NAMES = {
    4: "build", 5: "build_road", 10: "dice_roll", 11: "robber_placed",
    14: "steal", 15: "receive_cards", 20: "play_dev_card", 44: "end_turn",
    45: "game_over", 47: "receive_resources", 55: "receive_cards_multi",
    60: "discard", 66: "gain_achievement", 86: "monopoly_steal", 116: "trade",
}
PIECE_NAMES = {0: "road", 2: "settlement", 3: "city"}

# ---------------------------------------------------------------------------
# Balanced dice engine (ported from archive/tools/diceTracker.py — single source of truth)
# ---------------------------------------------------------------------------

DICE_TOTALS = list(range(2, 13))


@dataclass
class _SevenStreak:
    player: Optional[object] = None
    count: int = 0


class BalancedDiceEngine:
    """
    Reverse-engineered from the Colonist TypeScript DiceControllerBalanced.
    Models the deck as counts of remaining outcome 'cards' per total (2..12).
    """

    def __init__(
        self,
        players: List,
        minimum_cards_before_reshuffling: int = 13,
        probability_reduction_for_recently_rolled: float = 0.34,
        probability_reduction_for_seven_streaks: float = 0.4,
        maximum_recent_roll_memory: int = 5,
    ):
        self.players = players[:]
        self.number_of_players = len(players)
        self.minimum_cards_before_reshuffling = minimum_cards_before_reshuffling
        self.prob_reduction_recent = probability_reduction_for_recently_rolled
        self.prob_reduction_seven_streaks = probability_reduction_for_seven_streaks
        self.maximum_recent_roll_memory = maximum_recent_roll_memory
        self.deck_counts: Dict[int, int] = {t: 0 for t in DICE_TOTALS}
        self.cards_left: int = 0
        self.recent_rolls: List[int] = []
        self.recently_rolled_count: Dict[int, int] = {t: 0 for t in DICE_TOTALS}
        self.seven_streak = _SevenStreak(player=None, count=0)
        self.total_sevens_by_player: Dict = {}
        self.reshuffle()

    @staticmethod
    def standard_counts() -> Dict[int, int]:
        return {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

    def reshuffle(self) -> None:
        std = self.standard_counts()
        for t in DICE_TOTALS:
            self.deck_counts[t] = std[t]
        self.cards_left = 36

    def _init_total_sevens(self, player) -> None:
        if player not in self.total_sevens_by_player:
            self.total_sevens_by_player[player] = 0

    def _update_recent_window(self) -> None:
        if len(self.recent_rolls) <= self.maximum_recent_roll_memory:
            return
        oldest = self.recent_rolls.pop(0)
        self.recently_rolled_count[oldest] = max(0, self.recently_rolled_count[oldest] - 1)

    def _update_seven_rolls(self, player) -> None:
        self._init_total_sevens(player)
        self.total_sevens_by_player[player] += 1
        if self.seven_streak.player == player:
            self.seven_streak.count += 1
        else:
            self.seven_streak.player = player
            self.seven_streak.count = 1

    def _get_total_sevens_rolled(self) -> int:
        return sum(self.total_sevens_by_player.values())

    def _get_streak_adjustment_constant(self, player) -> float:
        if self.seven_streak.player is None:
            return 0.0
        sign = -1 if self.seven_streak.player == player else 1
        return self.prob_reduction_seven_streaks * self.seven_streak.count * sign

    def _get_seven_imbalance_adjustment(self, player) -> float:
        total_sevens = self._get_total_sevens_rolled()
        initialized_players = len(self.total_sevens_by_player)
        if initialized_players == 0 or total_sevens < initialized_players or total_sevens == 0:
            return 1.0
        player_sevens = self.total_sevens_by_player.get(player, 0)
        percentage_of_total = player_sevens / total_sevens
        ideal_percentage = 1.0 / initialized_players
        return 1.0 + ((ideal_percentage - percentage_of_total) / ideal_percentage)

    def _seven_probability_multiplier_for_player(self, player) -> float:
        if self.number_of_players < 2:
            return 1.0
        self._init_total_sevens(player)
        streak_adj = self._get_streak_adjustment_constant(player)
        imbalance_adj = self._get_seven_imbalance_adjustment(player)
        return max(0.0, min(2.0, 1.0 * imbalance_adj + streak_adj))

    def base_distribution(self) -> Dict[int, float]:
        if self.cards_left <= 0:
            self.reshuffle()
        return {t: self.deck_counts[t] / self.cards_left for t in DICE_TOTALS}

    def adjusted_distribution(self, player) -> Dict[int, float]:
        base = self.base_distribution()
        adjusted = {}
        for t in DICE_TOTALS:
            reduction = self.recently_rolled_count[t] * self.prob_reduction_recent
            adjusted[t] = base[t] * max(0.0, 1.0 - reduction)
        adjusted[7] *= self._seven_probability_multiplier_for_player(player)
        s = sum(adjusted.values())
        if s <= 0:
            sb = sum(base.values())
            return {t: (base[t] / sb if sb > 0 else 0.0) for t in DICE_TOTALS}
        return {t: adjusted[t] / s for t in DICE_TOTALS}

    def apply_roll(self, player, total: int) -> None:
        if total not in DICE_TOTALS:
            return
        if self.cards_left < self.minimum_cards_before_reshuffling:
            self.reshuffle()
        if self.deck_counts[total] <= 0:
            self.reshuffle()
        self.deck_counts[total] -= 1
        self.cards_left -= 1
        self.recent_rolls.append(total)
        self.recently_rolled_count[total] += 1
        self._update_recent_window()
        self._init_total_sevens(player)
        if total == 7:
            self._update_seven_rolls(player)

    def snapshot(self) -> Dict:
        return {
            "cards_left": self.cards_left,
            "deck_counts": dict(self.deck_counts),
            "recent_rolls": list(self.recent_rolls),
            "recently_rolled_count": dict(self.recently_rolled_count),
            "seven_streak_player": self.seven_streak.player,
            "seven_streak_count": self.seven_streak.count,
            "total_sevens_by_player": dict(self.total_sevens_by_player),
        }


def _engine_from_snapshot(players, snap) -> BalancedDiceEngine:
    """Reconstruct a BalancedDiceEngine from a snapshot dict."""
    eng = BalancedDiceEngine(players)
    eng.deck_counts = dict(snap["deck_counts"])
    eng.cards_left = snap["cards_left"]
    eng.recent_rolls = list(snap["recent_rolls"])
    eng.recently_rolled_count = dict(snap["recently_rolled_count"])
    eng.seven_streak.player = snap["seven_streak_player"]
    eng.seven_streak.count = snap["seven_streak_count"]
    eng.total_sevens_by_player = dict(snap["total_sevens_by_player"])
    return eng


# ---------------------------------------------------------------------------
# Hex geometry helpers
# ---------------------------------------------------------------------------

def axial_to_pixel(q, r, size=1.0):
    x = size * (SQRT3 * q + SQRT3 / 2.0 * r)
    y = size * (-1.5 * r)
    return x, y


def corner_pixel(x, y, z, size=1.0):
    """Pixel position of Colonist corner (x,y,z) = centroid of its 3 adjacent hexes."""
    if z == 0:
        hexes = [(x, y), (x, y - 1), (x + 1, y - 1)]
    else:
        hexes = [(x, y), (x, y + 1), (x - 1, y + 1)]
    pts = [axial_to_pixel(hx, hy, size) for hx, hy in hexes]
    return sum(p[0] for p in pts) / 3, sum(p[1] for p in pts) / 3



def build_positions(corner_states, edge_states):
    """Compute pixel positions for all corners and edges directly from initialState.

    Corner formula: corner(x,y,z) pixel = centroid of its 3 adjacent hexes.
    Edge formula:
      edge(x,y,z=0) endpoints: corner(x,y,0)      and corner(x,y-1,1)
      edge(x,y,z=1) endpoints: corner(x-1,y+1,0)  and corner(x,y-1,1)
      edge(x,y,z=2) endpoints: corner(x-1,y+1,0)  and corner(x,y,1)

    Returns:
        v_pos:     {corner_id (int) -> (px, py)}
        e_pos:     {edge_id   (int) -> ((x1,y1), (x2,y2))}
        e_corners: {edge_id   (int) -> (corner_id_a, corner_id_b)}
    """
    v_pos = {int(cid): corner_pixel(cs["x"], cs["y"], cs["z"])
             for cid, cs in corner_states.items()}
    pixel_to_vid = {v: k for k, v in v_pos.items()}

    EDGE_CORNERS = {
        0: lambda x, y: ((x, y, 0),          (x, y - 1, 1)),
        1: lambda x, y: ((x - 1, y + 1, 0),  (x, y - 1, 1)),
        2: lambda x, y: ((x - 1, y + 1, 0),  (x, y, 1)),
    }
    e_pos = {}
    e_corners = {}
    for eid_str, es in edge_states.items():
        x, y, z = es["x"], es["y"], es["z"]
        (ax, ay, az), (bx, by, bz) = EDGE_CORNERS[z](x, y)
        pa = corner_pixel(ax, ay, az)
        pb = corner_pixel(bx, by, bz)
        eid = int(eid_str)
        e_pos[eid] = (pa, pb)
        va, vb = pixel_to_vid.get(pa), pixel_to_vid.get(pb)
        if va is not None and vb is not None:
            e_corners[eid] = (va, vb)
    return v_pos, e_pos, e_corners


def compute_longest_road(player_color, edges_state, corners_state, e_corners):
    """Return the longest continuous road length for player_color.

    Roads are broken at corners occupied by opponent settlements/cities.
    Uses edge-visited DFS (an edge may not be traversed twice in one path).
    """
    adj = {}   # corner_id → [neighbour_corner_id, ...]
    for eid, (ca, cb) in e_corners.items():
        if edges_state.get(str(eid), {}).get("owner") == player_color:
            adj.setdefault(ca, []).append(cb)
            adj.setdefault(cb, []).append(ca)
    if not adj:
        return 0

    def passable(cid):
        o = corners_state.get(str(cid), {}).get("owner")
        return o is None or o == player_color

    def dfs(u, visited_edges):
        best = 0
        if not passable(u):
            return best
        for v in adj.get(u, []):
            e = (min(u, v), max(u, v))
            if e not in visited_edges:
                visited_edges.add(e)
                best = max(best, 1 + dfs(v, visited_edges))
                visited_edges.remove(e)
        return best

    return max(dfs(s, set()) for s in adj)


def hex_corners(cx, cy, size=1.0):
    return [
        (cx + size * math.cos(math.radians(30 + 60 * i)),
         cy + size * math.sin(math.radians(30 + 60 * i)))
        for i in range(6)
    ]


def pips_for_number(n):
    return {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}.get(n, 0)


def edge_endpoints(q, r, z):
    """Return the two corner pixel positions for port edge z of hex at axial (q, r).
    z selects which coastal edge: 0→corners(1,2), 1→corners(2,3), 2→corners(3,4).
    """
    cx, cy = axial_to_pixel(q, r)
    corners = hex_corners(cx, cy)
    z_to_corners = {0: (1, 2), 1: (2, 3), 2: (3, 4)}
    i, j = z_to_corners[z % 3]
    return corners[i], corners[j]

# ---------------------------------------------------------------------------
# Replay parsing
# ---------------------------------------------------------------------------

def parse_replay(path: Path):
    raw = json.loads(path.read_text())
    d   = raw["data"]
    ev_hist = d["eventHistory"]
    init    = ev_hist["initialState"]
    events  = ev_hist["events"]

    play_order = d["playOrder"]
    players    = {p["selectedColor"]: p["username"] for p in d["playerUserStates"]}
    game_id    = d["databaseGameId"]

    # --- Build tile map from initialState ---
    tile_data = {}   # hex_index -> {q, r, resource, number}
    for idx_str, t in init["mapState"]["tileHexStates"].items():
        tile_data[int(idx_str)] = {
            "q":        t["x"],
            "r":        t["y"],
            "resource": RESOURCE_NAMES.get(t["type"], "?"),
            "number":   t["diceNumber"],
        }

    # --- Port data from initialState ---
    port_data = []   # list of {q, r, z, type}
    for p in init["mapState"].get("portEdgeStates", {}).values():
        port_data.append({"q": p["x"], "r": p["y"], "z": p["z"], "type": p.get("type", 1)})

    # --- Corner / edge pixel positions directly from initialState ---
    v_pos, e_pos, e_corners = build_positions(
        init["mapState"].get("tileCornerStates", {}),
        init["mapState"].get("tileEdgeStates", {}),
    )

    # --- Initial game state ---
    def empty_state():
        s = {
            "turn": 0, "current_player": play_order[0],
            "action_state": None,
            "corners": {},
            "edges": {},
            "player_resources": {c: [] for c in play_order},
            "player_vp": {c: 0 for c in play_order},
            "player_vp_breakdown": {c: {} for c in play_order},
            "bank": {i: 19 for i in range(1, 6)},
            "robber_tile": init.get("mechanicRobberState", {}).get("locationTileIndex", 0),
            "player_dev_cards": {c: [] for c in play_order},
            "player_knights_played": {c: 0 for c in play_order},
            "player_road_length": {c: 0 for c in play_order},
            "largest_army_holder": None,
            "longest_road_holder": None,
            "completed_turns": 0,
        }
        # Seed corners/edges from initialState
        for vid, cd in init["mapState"].get("tileCornerStates", {}).items():
            s["corners"][vid] = dict(cd)
        for eid, ed in init["mapState"].get("tileEdgeStates", {}).items():
            s["edges"][eid] = dict(ed)
        # Seed player resources/VP from initialState
        for color_str, pd in init.get("playerStates", {}).items():
            color = int(color_str)
            if color in s["player_resources"]:
                s["player_resources"][color] = pd.get("resourceCards", {}).get("cards", [])
            if "victoryPointsState" in pd:
                s["player_vp_breakdown"][color].update(pd["victoryPointsState"])
                vps = s["player_vp_breakdown"][color]
                s["player_vp"][color] = (vps.get("0", 0) + 2 * vps.get("1", 0) + vps.get("2", 0))
        return s

    def recompute_vp(state):
        """Recompute all player VPs including Largest Army / Longest Road bonuses."""
        for c in play_order:
            vps = state["player_vp_breakdown"].get(c, {})
            state["player_vp"][c] = (
                vps.get("0", 0) + 2 * vps.get("1", 0) + vps.get("2", 0)
                + (2 if state["largest_army_holder"] == c else 0)
                + (2 if state["longest_road_holder"] == c else 0)
            )

    def update_largest_army(state, player):
        """Assign / transfer Largest Army after a knight play."""
        k       = state["player_knights_played"].get(player, 0)
        holder  = state["largest_army_holder"]
        if holder is None:
            if k >= 3:
                state["largest_army_holder"] = player
        elif player != holder and k > state["player_knights_played"].get(holder, 0):
            state["largest_army_holder"] = player

    def update_longest_road(state):
        """Assign / transfer Longest Road after any road/settlement change."""
        holder     = state["longest_road_holder"]
        road_lens  = state["player_road_length"]
        if holder is None:
            # Give to the first player in play order who reaches ≥5
            best_len, best_c = 0, None
            for c in play_order:
                if road_lens[c] >= 5 and road_lens[c] > best_len:
                    best_len, best_c = road_lens[c], c
            if best_c is not None:
                state["longest_road_holder"] = best_c
        else:
            holder_len = road_lens.get(holder, 0)
            for c in play_order:
                if c != holder and road_lens[c] > holder_len:
                    state["longest_road_holder"] = c
                    holder     = c
                    holder_len = road_lens[c]

    def deep_merge(base, diff):
        for k, v in diff.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_merge(base[k], v)
            else:
                base[k] = v
        return base

    def apply_event(state, event):
        sc = event.get("stateChange", {})

        map_changed = False
        for vid, cdata in sc.get("mapState", {}).get("tileCornerStates", {}).items():
            state["corners"].setdefault(vid, {})
            deep_merge(state["corners"][vid], cdata)
            map_changed = True
        for eid, edata in sc.get("mapState", {}).get("tileEdgeStates", {}).items():
            state["edges"].setdefault(eid, {})
            deep_merge(state["edges"][eid], edata)
            map_changed = True

        # Recompute road lengths whenever board topology changes
        if map_changed:
            for c in play_order:
                state["player_road_length"][c] = compute_longest_road(
                    c, state["edges"], state["corners"], e_corners)
            update_longest_road(state)

        for color_str, pdata in sc.get("playerStates", {}).items():
            color = int(color_str)
            if "resourceCards" in pdata:
                state["player_resources"][color] = pdata["resourceCards"].get("cards", [])
            if "victoryPointsState" in pdata:
                state["player_vp_breakdown"].setdefault(color, {}).update(pdata["victoryPointsState"])

        dev_state = sc.get("mechanicDevelopmentCardsState", {}).get("players", {})
        for color_str, ddata in dev_state.items():
            color = int(color_str)
            if color in state["player_dev_cards"] and "developmentCards" in ddata:
                state["player_dev_cards"][color] = ddata["developmentCards"].get("cards", [])

        for res_str, cnt in sc.get("bankState", {}).get("resourceCards", {}).items():
            state["bank"][int(res_str)] = cnt

        dice = sc.get("diceState", {})
        if dice.get("diceThrown"):
            state["last_dice"] = (dice.get("dice1", 0), dice.get("dice2", 0))

        cs = sc.get("currentState", {})
        prev_player = state["current_player"]
        if "currentTurnPlayerColor" in cs:
            state["current_player"] = cs["currentTurnPlayerColor"]
        if "actionState" in cs:
            state["action_state"] = cs["actionState"]
        if "completedTurns" in cs:
            state["completed_turns"] = cs["completedTurns"]
            state["turn"] = cs["completedTurns"]

        rs = sc.get("mechanicRobberState", {})
        if "locationTileIndex" in rs:
            state["robber_tile"] = rs["locationTileIndex"]

        # Parse human-readable actions from log; also track knight plays inline
        actions = []
        for _, log_entry in sc.get("gameLogState", {}).items():
            text = log_entry.get("text", {})
            if not isinstance(text, dict):
                continue
            ltype  = text.get("type")
            player = text.get("playerColor") or log_entry.get("from")

            # cardEnum 0 = knight in play-dev-card log events
            if ltype == 20 and text.get("cardEnum") == 0 and player in state["player_knights_played"]:
                state["player_knights_played"][player] += 1
                update_largest_army(state, player)

            if ltype == 44:
                actions.append({"type": "end_turn", "player": prev_player})
            elif ltype == 1:
                # Buy dev card: find which card was bought from mechanicDevelopmentCardsState
                bought = []
                p_str = str(player)
                if p_str in dev_state:
                    bought = dev_state[p_str].get("developmentCardsBoughtThisTurn", [])
                card_enum = bought[0] if bought else None
                actions.append({"type": "buy_dev_card", "player": player, "card_enum": card_enum})
            else:
                act = _parse_action(ltype, text, player, state)
                if act:
                    actions.append(act)

        recompute_vp(state)
        return actions

    def _parse_action(ltype, text, player, state):
        if ltype == 10:
            d1, d2 = text.get("firstDice", 0), text.get("secondDice", 0)
            return {"type": "dice_roll", "player": player, "dice": [d1, d2], "total": d1 + d2}
        if ltype == 4:
            piece = PIECE_NAMES.get(text.get("pieceEnum"), "?")
            return {"type": f"build_{piece}", "player": player}
        if ltype == 5:
            return {"type": "build_road", "player": player}
        if ltype == 11:
            ti = text.get("tileInfo", {})
            return {"type": "place_robber", "player": player,
                    "tile": state["robber_tile"],
                    "blocked": RESOURCE_NAMES.get(ti.get("resourceType"))}
        if ltype == 14:
            stolen = [RESOURCE_NAMES.get(c, f"?{c}") for c in text.get("cardEnums", [])]
            return {"type": "steal", "player": player, "stolen": stolen}
        if ltype == 20:
            card = DEV_CARD_NAMES.get(text.get("cardEnum"), f"dev_{text.get('cardEnum')}")
            return {"type": "play_dev_card", "player": player, "card": card}
        if ltype == 47:
            cards = [RESOURCE_NAMES.get(c, f"?{c}") for c in text.get("cardsToBroadcast", [])]
            return {"type": "receive_resources", "player": player, "resources": cards}
        if ltype == 55:
            cards = [RESOURCE_NAMES.get(c, f"?{c}") for c in text.get("cardEnums", [])]
            return {"type": "receive_cards", "player": player, "resources": cards}
        if ltype == 60:
            if player is None:
                return None  # generic "players must discard" broadcast, not player-specific
            return {"type": "discard", "player": player}
        if ltype == 86:
            return {"type": "monopoly", "player": player,
                    "resource": RESOURCE_NAMES.get(text.get("cardEnum")),
                    "amount": text.get("amountStolen", 0)}
        if ltype == 116:
            give = [RESOURCE_NAMES.get(c, f"?{c}") for c in text.get("givenCardEnums", [])]
            recv = [RESOURCE_NAMES.get(c, f"?{c}") for c in text.get("receivedCardEnums", [])]
            return {"type": "trade", "player": player, "give": give, "receive": recv}
        if ltype == 45:
            return {"type": "game_over", "player": player}
        return None

    # --- Build timeline ---
    state    = empty_state()
    timeline = []  # list of {action, state_snapshot}

    for i, event in enumerate(events):
        state_before = deepcopy(state)
        actions      = apply_event(state, event)

        for act in actions:
            if act["type"] in ("game_init", "connected"):
                continue
            p = act["player"]
            record = {
                "event_idx":        i,
                "turn":             state["turn"],
                "player":           p,
                "player_name":      players.get(p, f"p{p}"),
                "action":           act,
                "resources_before": list(state_before["player_resources"].get(p, [])),
                "resources_after":  list(state["player_resources"].get(p, [])),
                "vp":               state["player_vp"].get(p, 0),
                "robber_tile":      state["robber_tile"],
                "state":            deepcopy(state),
            }
            timeline.append(record)

    return {
        "game_id":    game_id,
        "players":    players,
        "play_order": play_order,
        "tile_data":  tile_data,
        "port_data":  port_data,
        "v_pos":      v_pos,
        "e_pos":      e_pos,
        "timeline":   timeline,
    }

# ---------------------------------------------------------------------------
# Board drawing
# ---------------------------------------------------------------------------

def _draw_robber(ax, cx, cy, size=1.0):
    """Draw a Catan-style pawn/robber piece centered on hex (cx, cy)."""
    sc = size
    # Anchor the piece slightly above hex center so it clears the number token
    px, py = cx, cy + 0.18 * sc

    c_dark  = "#484855"
    c_mid   = "#70707e"
    c_light = "#9898aa"
    c_hi    = "#bcbccc"
    outline = "#28283050"     # semi-transparent dark outline

    # Drop shadow
    ax.add_patch(mpatches.Ellipse(
        (px + 0.035 * sc, py - 0.29 * sc), 0.40 * sc, 0.09 * sc,
        facecolor="#000", alpha=0.20, edgecolor="none", zorder=5))

    # Base disc
    ax.add_patch(mpatches.Ellipse(
        (px, py - 0.27 * sc), 0.38 * sc, 0.11 * sc,
        facecolor=c_mid, edgecolor=c_dark, lw=0.7, zorder=6))

    # Body (trapezoid — wider at base, narrower toward neck)
    ax.add_patch(mpatches.Polygon([
        [px - 0.165 * sc, py - 0.27 * sc],
        [px + 0.165 * sc, py - 0.27 * sc],
        [px + 0.09  * sc, py + 0.06 * sc],
        [px - 0.09  * sc, py + 0.06 * sc],
    ], facecolor=c_mid, edgecolor=c_dark, lw=0.7, zorder=6))

    # Left-side body highlight (bevel)
    ax.add_patch(mpatches.Polygon([
        [px - 0.165 * sc, py - 0.27 * sc],
        [px - 0.06  * sc, py - 0.27 * sc],
        [px - 0.01  * sc, py + 0.06 * sc],
        [px - 0.09  * sc, py + 0.06 * sc],
    ], facecolor=c_light, edgecolor="none", alpha=0.55, zorder=7))

    # Neck
    ax.add_patch(mpatches.Polygon([
        [px - 0.07  * sc, py + 0.06 * sc],
        [px + 0.07  * sc, py + 0.06 * sc],
        [px + 0.055 * sc, py + 0.17 * sc],
        [px - 0.055 * sc, py + 0.17 * sc],
    ], facecolor=c_mid, edgecolor=c_dark, lw=0.7, zorder=6))

    # Head
    ax.add_patch(plt.Circle(
        (px, py + 0.285 * sc), 0.12 * sc,
        facecolor=c_hi, edgecolor=c_dark, lw=0.8, zorder=8))

    # Head gleam (top-left highlight)
    ax.add_patch(plt.Circle(
        (px - 0.033 * sc, py + 0.325 * sc), 0.042 * sc,
        facecolor="white", edgecolor="none", alpha=0.52, zorder=9))


def _draw_settlement(ax, cx, cy, color, ec):
    """Draw a Catan-style house settlement piece centered at (cx, cy)."""
    x, y = cx, cy

    # Pentagon: rectangular body + peaked roof
    body_pts = [
        [x - 0.22, y - 0.24],   # bottom-left
        [x + 0.22, y - 0.24],   # bottom-right
        [x + 0.22, y + 0.02],   # wall top-right
        [x,        y + 0.34],   # roof peak
        [x - 0.22, y + 0.02],   # wall top-left
    ]
    ax.add_patch(mpatches.Polygon(body_pts, facecolor=color, edgecolor=ec, lw=1.2, zorder=6))

    # Right-side shading (dark overlay for 3-D bevel)
    shade_pts = [
        [x + 0.04, y - 0.24],
        [x + 0.22, y - 0.24],
        [x + 0.22, y + 0.02],
        [x,        y + 0.34],
        [x + 0.04, y + 0.22],
    ]
    ax.add_patch(mpatches.Polygon(shade_pts, facecolor="black", edgecolor="none",
                                  alpha=0.22, zorder=7))

    # Door
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - 0.06, y - 0.24), 0.12, 0.14,
        boxstyle="round,pad=0.01",
        facecolor="black", edgecolor="none", alpha=0.38, zorder=7))


def _draw_city(ax, cx, cy, color, ec):
    """Draw a Catan-style two-section city building piece centered at (cx, cy)."""
    x, y = cx, cy

    # Tall main tower (right section)
    tower_pts = [
        [x - 0.04, y - 0.32],
        [x + 0.26, y - 0.32],
        [x + 0.26, y + 0.20],
        [x - 0.04, y + 0.20],
    ]
    ax.add_patch(mpatches.Polygon(tower_pts, facecolor=color, edgecolor=ec, lw=1.2, zorder=6))

    # Shorter side annex (left section)
    annex_pts = [
        [x - 0.28, y - 0.32],
        [x - 0.04, y - 0.32],
        [x - 0.04, y + 0.04],
        [x - 0.28, y + 0.04],
    ]
    ax.add_patch(mpatches.Polygon(annex_pts, facecolor=color, edgecolor=ec, lw=1.2, zorder=6))

    # Right-side shading on tower
    shade_pts = [
        [x + 0.12, y - 0.32],
        [x + 0.26, y - 0.32],
        [x + 0.26, y + 0.20],
        [x + 0.12, y + 0.20],
    ]
    ax.add_patch(mpatches.Polygon(shade_pts, facecolor="black", edgecolor="none",
                                  alpha=0.22, zorder=7))

    # Window on tower face
    ax.add_patch(mpatches.FancyBboxPatch(
        (x + 0.04, y + 0.04), 0.10, 0.10,
        boxstyle="round,pad=0.01",
        facecolor="black", edgecolor="none", alpha=0.28, zorder=7))

    # Door on annex
    ax.add_patch(mpatches.FancyBboxPatch(
        (x - 0.20, y - 0.32), 0.10, 0.14,
        boxstyle="round,pad=0.01",
        facecolor="black", edgecolor="none", alpha=0.30, zorder=7))


def draw_board(state, tile_data, port_data, play_order, players, v_pos, e_pos, highlight_action=None):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect("equal")
    ax.axis("off")
    size = 1.0
    robber_tile = state.get("robber_tile", -1)

    # --- Hexes ---
    for hex_idx, td in tile_data.items():
        q, r   = td["q"], td["r"]
        cx, cy = axial_to_pixel(q, r, size)
        corners = hex_corners(cx, cy, size)
        xs = [c[0] for c in corners] + [corners[0][0]]
        ys = [c[1] for c in corners] + [corners[0][1]]

        fill_color = RESOURCE_COLORS.get(td["resource"], "#cccccc")
        edge_color = "#555" if hex_idx != robber_tile else "#ff0000"
        lw         = 0.8   if hex_idx != robber_tile else 2.5
        ax.fill(xs, ys, color=fill_color, ec=edge_color, lw=lw, zorder=1)

        # Number token circle
        n = td["number"]
        if n:
            pips  = pips_for_number(n)
            tfill = "#ffdddd" if pips == 5 else "white"
            circle = plt.Circle((cx, cy), 0.30, color=tfill, ec="#333", lw=0.5, zorder=2)
            ax.add_patch(circle)
            # Shift number+pips up as a group so the block is vertically centered
            # in the circle (group center sits at cy rather than cy-0.08).
            n_y   = cy + 0.07   # number center
            pip_y = cy - 0.14   # pip row  (n_y - 0.21 spacing)
            ax.text(cx, n_y, str(n), ha="center", va="center",
                    fontsize=7, fontweight="bold", color="#333", zorder=3)
            for p_i in range(pips):
                dot_x = cx + (p_i - (pips - 1) / 2) * 0.09
                ax.plot(dot_x, pip_y, "o", ms=2, color="#c0392b", zorder=3)
        # Robber piece
        if hex_idx == robber_tile:
            _draw_robber(ax, cx, cy, size)

    # --- Ports ---
    PORT_LABELS = {1: "3:1\nAny", 2: "2:1\nWood", 3: "2:1\nBrick",
                   4: "2:1\nSheep", 5: "2:1\nWheat", 6: "2:1\nOre"}
    PORT_COLORS_MAP = {1: "#aaaaaa", 2: "#4a7c3f", 3: "#c44b2b",
                       4: "#8cc84b", 5: "#f0c030", 6: "#8a8a8a"}
    for port in port_data:
        try:
            (px1, py1), (px2, py2) = edge_endpoints(port["q"], port["r"], port["z"])
        except Exception:
            continue
        ptype = port["type"]
        color = PORT_COLORS_MAP.get(ptype, "#aaa")
        label = PORT_LABELS.get(ptype, "?")
        mx, my = (px1 + px2) / 2, (py1 + py2) / 2
        dist = math.hypot(mx, my)
        nx, ny = (mx / dist, my / dist) if dist > 0 else (0, 1)
        lx, ly = mx + nx * 0.7, my + ny * 0.7
        ax.plot([lx, px1], [ly, py1], color=color, lw=2.5, solid_capstyle="round", zorder=3)
        ax.plot([lx, px2], [ly, py2], color=color, lw=2.5, solid_capstyle="round", zorder=3)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=6.5,
                fontweight="bold", color="white", multialignment="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc=color, ec="white", lw=1.2, alpha=0.95))

    # --- Roads ---
    for col_e_str, edata in state.get("edges", {}).items():
        col_e = int(col_e_str)
        owner = edata.get("owner")
        if owner and col_e in e_pos:
            (x1, y1), (x2, y2) = e_pos[col_e]
            color = PLAYER_MCOLORS.get(owner, "gray")
            ax.plot([x1, x2], [y1, y2], color=color, lw=4,
                    solid_capstyle="round", zorder=5)
            ax.plot([x1, x2], [y1, y2], color=PLAYER_EDGE_COLORS.get(owner, "black"),
                    lw=4.5, solid_capstyle="round", zorder=4)

    # --- Settlements & Cities ---
    for col_v_str, cdata in state.get("corners", {}).items():
        col_v = int(col_v_str)
        owner = cdata.get("owner")
        btype = cdata.get("buildingType", 0)
        if owner and col_v in v_pos:
            x, y  = v_pos[col_v]
            color = PLAYER_MCOLORS.get(owner, "gray")
            ec    = PLAYER_EDGE_COLORS.get(owner, "black")
            if btype == 1:
                _draw_settlement(ax, x, y, color, ec)
            elif btype == 2:
                _draw_city(ax, x, y, color, ec)

    # --- Legend ---
    legend_handles = []
    for color in play_order:
        name  = players.get(color, f"p{color}")
        patch = mpatches.Patch(color=PLAYER_MCOLORS.get(color, "gray"),
                               ec=PLAYER_EDGE_COLORS.get(color, "black"),
                               label=f"Player {color}: {name}",
                               linewidth=1)
        legend_handles.append(patch)
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7,
              framealpha=0.85, edgecolor="#aaa")

    margin = 0.3
    ax.set_xlim(-4.5 - margin, 4.5 + margin)
    ax.set_ylim(-4.5 - margin, 4.5 + margin)
    fig.tight_layout(pad=0.2)
    return fig

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

def action_label(rec):
    act  = rec["action"]
    name = rec["player_name"]
    t    = act["type"]
    if t == "dice_roll":
        return f"🎲 {name} rolls {act['dice'][0]}+{act['dice'][1]}={act['total']}"
    if t == "build_settlement":
        return f"🏠 {name} builds settlement"
    if t == "build_city":
        return f"🏙 {name} builds city"
    if t == "build_road":
        return f"🛣 {name} builds road"
    if t == "place_robber":
        return f"🏴 {name} moves robber → tile {act.get('tile')} ({act.get('blocked','')})"
    if t == "steal":
        return f"🗡 {name} steals {act.get('stolen', [])}"
    if t == "buy_dev_card":
        card_enum = act.get("card_enum")
        card_name = _DEV_CARD_NAMES.get(card_enum, f"dev_{card_enum}") if card_enum is not None else "?"
        return f"🃏 {name} buys dev card ({card_name})"
    if t == "play_dev_card":
        return f"🃏 {name} plays {act.get('card')}"
    if t == "end_turn":
        return f"↩ {name} ends turn"
    if t == "receive_resources":
        return f"📦 {name} receives {act.get('resources', [])}"
    if t == "receive_cards":
        return f"📦 {name} receives {act.get('resources', [])}"
    if t == "discard":
        return f"🗑 {name} discards"
    if t == "monopoly":
        return f"👑 {name} monopoly: {act.get('resource')} ×{act.get('amount')}"
    if t == "trade":
        return f"🔄 {name} trades {act.get('give')} → {act.get('receive')}"
    if t == "game_over":
        return f"🏆 {name} wins!"
    return f"  {name}: {t}"


def resource_bar(resources, player_colors, play_order, players):
    """Render per-player resource counts as a compact table."""
    RNAMES = {1: "🪵", 2: "🧱", 3: "🐑", 4: "🌾", 5: "⛏"}
    cols = st.columns(len(play_order))
    for i, color in enumerate(play_order):
        name  = players.get(color, f"p{color}")
        cards = resources.get(color, [])
        counts = {}
        for c in cards:
            counts[c] = counts.get(c, 0) + 1
        with cols[i]:
            st.markdown(f"**{name}**")
            card_str = "  ".join(f"{RNAMES.get(r,'?')}×{n}" for r, n in sorted(counts.items())) or "—"
            st.markdown(card_str)


# ---------------------------------------------------------------------------
# Visual card rendering
# ---------------------------------------------------------------------------

_CARD_THEMES = {
    1: {"bg": "linear-gradient(135deg,#5fa84d 0%,#3a6c2f 100%)", "border": "#2a5020", "icon": "🌲"},
    2: {"bg": "linear-gradient(135deg,#d4603c 0%,#9e3820 100%)", "border": "#7a2e18", "icon": "🧱"},
    3: {"bg": "linear-gradient(135deg,#a0d45c 0%,#6aa832 100%)", "border": "#4a8822", "icon": "🐑"},
    4: {"bg": "linear-gradient(135deg,#f5cc40 0%,#c89010 100%)", "border": "#9a7000", "icon": "🌾"},
    5: {"bg": "linear-gradient(135deg,#8ab0be 0%,#5a7a8a 100%)", "border": "#3a5a6a", "icon": "⛏"},
}
_DEV_BORDER = "#4a1a8a"


def _card_html(bg: str, border: str, icon: str, count: int) -> str:
    """HTML for one card stack (resource or dev card)."""
    W, H = 34, 52

    stack = ""
    if count > 1:
        for off in (4, 2):
            stack += (
                f"<div style='position:absolute;top:{off}px;left:{off}px;"
                f"width:{W}px;height:{H}px;border-radius:5px;"
                f"background:{bg};border:1.5px solid {border};'></div>"
            )

    badge = ""
    if count > 1:
        badge = (
            f"<div style='position:absolute;top:-5px;right:-5px;min-width:17px;height:17px;"
            f"border-radius:9px;background:#1565c0;border:2px solid white;color:white;"
            f"font-size:10px;font-weight:700;display:flex;align-items:center;"
            f"justify-content:center;padding:0 3px;box-sizing:border-box;z-index:5;'>"
            f"{count}</div>"
        )

    is_svg = icon.lstrip().startswith("<")
    if is_svg:
        icon_div = (
            f"<div style='position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);'>"
            f"{icon}</div>"
        )
    else:
        icon_div = (
            f"<div style='position:absolute;top:50%;left:50%;transform:translate(-50%,-46%);"
            f"font-size:19px;line-height:1;'>{icon}</div>"
        )

    card = (
        f"<div style='position:relative;width:{W}px;height:{H}px;border-radius:5px;"
        f"background:{bg};border:1.5px solid {border};"
        f"box-shadow:0 2px 6px rgba(0,0,0,0.55);'>"
        # inner highlight strip at top
        f"<div style='position:absolute;top:3px;left:3px;right:3px;height:14px;"
        f"border-radius:3px 3px 0 0;background:rgba(255,255,255,0.22);'></div>"
        f"{icon_div}"
        f"</div>"
    )

    extra = 6 if count > 1 else 0
    return (
        f"<div style='display:inline-block;position:relative;margin-right:8px;"
        f"margin-bottom:4px;vertical-align:top;"
        f"width:{W + extra}px;height:{H + extra}px;'>"
        f"{stack}{card}{badge}</div>"
    )


# dev card enum → display name (verified from replay data)
_DEV_CARD_NAMES = {11: "knight", 12: "victory_point", 13: "monopoly", 14: "road_building", 15: "year_of_plenty"}
_DEV_BG     = "linear-gradient(135deg,#9c5fd4 0%,#6a2faa 100%)"
_DEV_BORDER = "#4a1a8a"

# Inline SVG icons for each dev card type
_DEV_CARD_ICONS = {
    "knight": (
        '<svg width="22" height="22" viewBox="0 0 22 22" xmlns="http://www.w3.org/2000/svg">'
        # Helmet dome
        '<path d="M7 10.5Q7 4 11 3Q15 4 15 10.5L15 12H7Z" fill="rgba(225,215,255,0.85)"/>'
        # Visor bar
        '<rect x="7.2" y="11.2" width="7.6" height="2.5" rx="0.6" fill="rgba(200,185,248,0.75)"/>'
        # Visor eye-slit
        '<rect x="9" y="11.8" width="4" height="0.9" rx="0.3" fill="rgba(110,70,190,0.55)"/>'
        # Neck guard
        '<path d="M8.8 13.7L8.2 17H13.8L13.2 13.7Z" fill="rgba(215,205,252,0.8)"/>'
        # Base plate
        '<rect x="6.2" y="17" width="9.6" height="2" rx="1" fill="rgba(225,215,255,0.85)"/>'
        '</svg>'
    ),
    "victory_point": (
        '<svg width="22" height="22" viewBox="0 0 22 22" xmlns="http://www.w3.org/2000/svg">'
        # Trophy cup body
        '<path d="M7.5 4.5h7v5.5c0 1.93-1.57 3.5-3.5 3.5s-3.5-1.57-3.5-3.5V4.5z" fill="rgba(255,215,60,0.92)"/>'
        # Left handle
        '<path d="M7.5 6.5C6 6.5 5 7.5 5 9s1 2.5 2.5 2.5" fill="none" stroke="rgba(255,215,60,0.92)" stroke-width="1.5" stroke-linecap="round"/>'
        # Right handle
        '<path d="M14.5 6.5C16 6.5 17 7.5 17 9s-1 2.5-2.5 2.5" fill="none" stroke="rgba(255,215,60,0.92)" stroke-width="1.5" stroke-linecap="round"/>'
        # Stem
        '<rect x="10.3" y="13.5" width="1.4" height="2.5" fill="rgba(255,215,60,0.8)"/>'
        # Base
        '<rect x="7.5" y="16" width="7" height="1.8" rx="0.8" fill="rgba(255,215,60,0.92)"/>'
        # +1 inside cup
        '<text x="11" y="11.2" text-anchor="middle" font-size="4.8" font-weight="bold"'
        ' fill="rgba(120,70,200,0.95)" font-family="sans-serif">+1</text>'
        '</svg>'
    ),
    "year_of_plenty": (
        '<svg width="22" height="22" viewBox="0 0 22 22" xmlns="http://www.w3.org/2000/svg">'
        # Three resource dots in a triangle (wheat/wood/ore colors)
        '<circle cx="11" cy="5.5" r="3.8" fill="rgba(240,185,50,0.92)"/>'
        '<circle cx="6.5" cy="14" r="3.8" fill="rgba(80,165,55,0.92)"/>'
        '<circle cx="15.5" cy="14" r="3.8" fill="rgba(150,115,200,0.88)"/>'
        '</svg>'
    ),
    "road_building": (
        '<svg width="22" height="22" viewBox="0 0 22 22" xmlns="http://www.w3.org/2000/svg">'
        # Road bar 1
        '<rect x="3.5" y="5.5" width="15" height="4" rx="1.5" fill="rgba(220,210,250,0.88)"/>'
        '<rect x="6" y="7.1" width="2.5" height="0.9" rx="0.3" fill="rgba(120,80,200,0.45)"/>'
        '<rect x="9.75" y="7.1" width="2.5" height="0.9" rx="0.3" fill="rgba(120,80,200,0.45)"/>'
        '<rect x="13.5" y="7.1" width="2.5" height="0.9" rx="0.3" fill="rgba(120,80,200,0.45)"/>'
        # Road bar 2
        '<rect x="3.5" y="12.5" width="15" height="4" rx="1.5" fill="rgba(220,210,250,0.88)"/>'
        '<rect x="6" y="14.1" width="2.5" height="0.9" rx="0.3" fill="rgba(120,80,200,0.45)"/>'
        '<rect x="9.75" y="14.1" width="2.5" height="0.9" rx="0.3" fill="rgba(120,80,200,0.45)"/>'
        '<rect x="13.5" y="14.1" width="2.5" height="0.9" rx="0.3" fill="rgba(120,80,200,0.45)"/>'
        '</svg>'
    ),
    "monopoly": (
        '<svg width="22" height="22" viewBox="0 0 22 22" xmlns="http://www.w3.org/2000/svg">'
        # Back card
        '<rect x="3" y="5" width="10" height="13" rx="1.5" fill="rgba(255,255,255,0.22)"'
        ' stroke="rgba(255,255,255,0.4)" stroke-width="0.8"/>'
        # Middle card
        '<rect x="5.5" y="3" width="10" height="13" rx="1.5" fill="rgba(255,255,255,0.42)"'
        ' stroke="rgba(255,255,255,0.58)" stroke-width="0.8"/>'
        # Front card
        '<rect x="8" y="1" width="10" height="13" rx="1.5" fill="rgba(255,255,255,0.72)"'
        ' stroke="rgba(255,255,255,0.9)" stroke-width="0.8"/>'
        # Star on front card
        '<text x="13" y="9" text-anchor="middle" dominant-baseline="middle"'
        ' font-size="7.5" fill="rgba(110,60,195,0.88)">★</text>'
        '</svg>'
    ),
}


_KNIGHT_PANEL_ICON = (
    '<svg width="12" height="16" viewBox="0 0 12 16" xmlns="http://www.w3.org/2000/svg">'
    '<ellipse cx="6" cy="14.5" rx="4.6" ry="1.5" fill="currentColor" opacity="0.5"/>'
    '<path d="M3.5 14 Q3.5 8.5 6 8.5 Q8.5 8.5 8.5 14Z" fill="currentColor"/>'
    '<circle cx="6" cy="5.5" r="3.8" fill="currentColor"/>'
    '</svg>'
)
_ROAD_PANEL_ICON = (
    '<svg width="17" height="11" viewBox="0 0 17 11" xmlns="http://www.w3.org/2000/svg">'
    '<path d="M0.8 10.2 C3.5 1 13.5 1 16.2 10.2" fill="none" '
    'stroke="currentColor" stroke-width="2.3" stroke-linecap="round"/>'
    '</svg>'
)


def render_player_panel_html(name: str, vp: int, cards: list, dev_cards: list,
                              settlements: list, cities: list, roads: list,
                              hex_color: str, knights_played: int = 0,
                              road_length: int = 0, has_largest_army: bool = False,
                              has_longest_road: bool = False) -> str:
    """Full HTML block for one player's row in the Player State panel."""
    # Resource card stacks
    res_counts: dict = {}
    for c in cards:
        res_counts[c] = res_counts.get(c, 0) + 1

    res_html = "".join(
        _card_html(_CARD_THEMES[r]["bg"], _CARD_THEMES[r]["border"], _CARD_THEMES[r]["icon"], n)
        for r, n in sorted(res_counts.items()) if r in _CARD_THEMES
    )

    # Dev card stacks (group by type name)
    dev_counts: dict = {}
    for enum in dev_cards:
        name_key = _DEV_CARD_NAMES.get(enum, f"dev_{enum}")
        dev_counts[name_key] = dev_counts.get(name_key, 0) + 1

    dev_html = "".join(
        _card_html(_DEV_BG, _DEV_BORDER, _DEV_CARD_ICONS.get(k, "🃏"), n)
        for k, n in sorted(dev_counts.items())
    )

    cards_html = res_html + dev_html
    if not cards_html:
        cards_html = "<span style='color:#aaa;font-size:11px;font-style:italic;'>—</span>"

    _GOLD = "#f5c518"
    _GRAY = "#888888"
    army_color = _GOLD if has_largest_army else _GRAY
    road_color  = _GOLD if has_longest_road  else _GRAY

    # Use dark text on light player colors (e.g. white player)
    try:
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        vp_text = "black" if 0.299 * r + 0.587 * g + 0.114 * b > 180 else "white"
    except Exception:
        vp_text = "white"

    return (
        f"<div style='border-left:4px solid {hex_color};padding:7px 10px 8px 10px;"
        f"margin-bottom:10px;background:rgba(255,255,255,0.03);border-radius:0 6px 6px 0;'>"
        # header row: name + VP badge + piece counts + knight + road
        f"<div style='margin-bottom:6px;display:flex;align-items:center;flex-wrap:wrap;gap:4px;'>"
        f"<b style='font-size:13px;'>{name}</b>"
        f"<span style='background:{hex_color};color:{vp_text};border-radius:4px;padding:1px 7px;"
        f"font-size:11px;font-weight:700;'>VP {vp}</span>"
        f"<span style='color:#aaa;font-size:11px;'>🏠×{len(settlements)}"
        f"&nbsp;🏙×{len(cities)}&nbsp;🛣×{len(roads)}</span>"
        # Knight count (gold when holding Largest Army)
        f"<span style='display:inline-flex;align-items:center;gap:2px;"
        f"color:{army_color};font-size:11px;font-weight:600;' "
        f"title='Knights played{'  ·  Largest Army' if has_largest_army else ''}'>"
        f"<span style='display:inline-flex;vertical-align:middle;'>{_KNIGHT_PANEL_ICON}</span>"
        f"&thinsp;{knights_played}</span>"
        # Road length (gold when holding Longest Road)
        f"<span style='display:inline-flex;align-items:center;gap:2px;"
        f"color:{road_color};font-size:11px;font-weight:600;' "
        f"title='Longest road{'  ·  Longest Road' if has_longest_road else ''}'>"
        f"<span style='display:inline-flex;vertical-align:middle;'>{_ROAD_PANEL_ICON}</span>"
        f"&thinsp;{road_length}</span>"
        f"</div>"
        # card strip
        f"<div style='display:flex;flex-wrap:wrap;align-items:flex-end;min-height:56px;'>"
        f"{cards_html}</div>"
        f"</div>"
    )


def _render_dice_chart(snap, play_order, players) -> bytes:
    """Render a compact balanced-dice distribution chart as PNG bytes."""
    eng = _engine_from_snapshot(play_order, snap)
    base = eng.base_distribution()
    n = len(play_order)

    fig, axes = plt.subplots(1, n, figsize=(4.6, 1.9), sharey=True,
                              gridspec_kw={"wspace": 0.06},
                              facecolor="#0e1117")
    if n == 1:
        axes = [axes]

    max_prob = max(
        max(eng.adjusted_distribution(c).values()) for c in play_order
    )

    for ax, color in zip(axes, play_order):
        adj = eng.adjusted_distribution(color)
        hex_col = PLAYER_MCOLORS.get(color, "#888888")

        # Bar fill: player color, slightly desaturated
        ax.set_facecolor("#0e1117")
        bar_vals = [adj[t] for t in DICE_TOTALS]
        ax.bar(DICE_TOTALS, bar_vals, color=hex_col, edgecolor="none",
               width=0.72, alpha=0.85, zorder=2)

        # Base distribution reference line (gray)
        ax.plot(DICE_TOTALS, [base[t] for t in DICE_TOTALS],
                color="#888888", lw=0.8, alpha=0.55, marker=".", ms=2, zorder=3)

        # Percent labels on top of every bar
        for t, v in zip(DICE_TOTALS, bar_vals):
            ax.text(t, v + max_prob * 0.03, f"{v*100:.0f}%",
                    ha="center", va="bottom", fontsize=4.5,
                    color="#cccccc", zorder=4)

        ax.set_xticks(DICE_TOTALS)
        ax.tick_params(axis="x", labelsize=5.5, colors="#aaaaaa", length=2)
        ax.tick_params(axis="y", labelsize=4.5, colors="#888888", length=2)
        ax.set_ylim(0, max_prob * 1.28)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))
        ax.set_title(players.get(color, str(color)), fontsize=6.5, pad=3,
                     color=hex_col, fontweight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # Footer: deck state + recent rolls
    recent = snap["recent_rolls"]
    recent_str = "  ".join(str(r) for r in recent) if recent else "—"
    cards_left = snap["cards_left"]
    fig.text(0.5, 0.0, f"deck {cards_left}/36  ·  recent: {recent_str}",
             ha="center", va="bottom", fontsize=5, color="#666666")

    fig.tight_layout(pad=0.5, rect=[0, 0.06, 1, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _board_key(state):
    """Hashable key that changes only when the visible board changes."""
    built = tuple(sorted(
        (k, v.get("owner"), v.get("buildingType", 0))
        for k, v in state["corners"].items() if v.get("owner")
    ))
    roads = tuple(sorted(
        (k, v.get("owner"))
        for k, v in state["edges"].items() if v.get("owner")
    ))
    return (built, roads, state.get("robber_tile"))


def main():
    st.set_page_config(page_title="Catan Replay Viewer", layout="wide")
    st.title("Catan Replay Viewer")

    # --- Replay selector ---
    replay_files = sorted(REPLAYS_DIR.glob("*.json"))
    if not replay_files:
        st.error(f"No replays found in {REPLAYS_DIR}. Download some first.")
        return

    file_names = [f.name for f in replay_files]
    chosen = st.sidebar.selectbox("Select replay", file_names)
    replay_path = REPLAYS_DIR / chosen

    # --- Parse + pre-render (cached per file) ---
    @st.cache_data
    def cached_parse(path_str):
        data = parse_replay(Path(path_str))
        # Pre-render all unique board states to PNG bytes so navigation is instant.
        # Many consecutive steps share the same board (dice rolls, trades, etc.),
        # so we only re-render when the visible board actually changes.
        prev_key = None
        prev_img = None
        for rec in data["timeline"]:
            key = _board_key(rec["state"])
            if key != prev_key:
                fig = draw_board(
                    rec["state"],
                    data["tile_data"], data["port_data"],
                    data["play_order"], data["players"],
                    data["v_pos"], data["e_pos"],
                )
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=95, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                prev_img = buf.read()
                prev_key = key
            rec["board_img"] = prev_img

        # Simulate balanced dice engine forward to snapshot state at each step
        dice_engine = BalancedDiceEngine(data["play_order"])
        for rec in data["timeline"]:
            act = rec["action"]
            if act["type"] == "dice_roll":
                dice_engine.apply_roll(act["player"], act["total"])
            rec["dice_snapshot"] = dice_engine.snapshot()

        return data

    data = cached_parse(str(replay_path))
    timeline   = data["timeline"]
    tile_data  = data["tile_data"]
    port_data  = data["port_data"]
    play_order = data["play_order"]
    players    = data["players"]
    game_id    = data["game_id"]
    v_pos      = data["v_pos"]
    e_pos      = data["e_pos"]

    if not timeline:
        st.warning("No actions found in this replay.")
        return

    st.sidebar.markdown(f"**Game ID:** {game_id}")
    st.sidebar.markdown(f"**Players:** {', '.join(players.values())}")
    st.sidebar.markdown(f"**Actions:** {len(timeline)}")

    # --- Step state: single source of truth is the slider's session state key ---
    slider_key = f"slider_{chosen}"
    if slider_key not in st.session_state:
        st.session_state[slider_key] = 0

    step = st.session_state[slider_key]

    # Precompute turn boundaries: indices where the turn number increases
    turn_boundaries = [
        i for i in range(1, len(timeline))
        if timeline[i]["turn"] > timeline[i - 1]["turn"]
    ]

    # Nav buttons: «  ‹  ›  »  — all write to slider_key so slider stays in sync
    col_tprev, col_prev, col_next, col_tnext = st.sidebar.columns([1, 1, 1, 1])
    with col_tprev:
        if st.button("«", key="turn_prev"):
            targets = [i for i in turn_boundaries if i < step]
            if targets:
                st.session_state[slider_key] = targets[-1]
                st.rerun()
    with col_prev:
        if st.button("‹", key="prev") and step > 0:
            st.session_state[slider_key] -= 1
            st.rerun()
    with col_next:
        if st.button("›", key="next") and step < len(timeline) - 1:
            st.session_state[slider_key] += 1
            st.rerun()
    with col_tnext:
        if st.button("»", key="turn_next"):
            targets = [i for i in turn_boundaries if i > step]
            if targets:
                st.session_state[slider_key] = targets[0]
                st.rerun()

    # Slider reads/writes slider_key directly — no separate sync needed
    st.sidebar.slider("Step", 0, len(timeline) - 1, key=slider_key)
    step = st.session_state[slider_key]
    rec  = timeline[step]

    # --- Action log sidebar ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Action log**")
    log_start = max(0, step - 8)
    for i in range(log_start, min(len(timeline), step + 5)):
        label = action_label(timeline[i])
        if i == step:
            st.sidebar.markdown(f"**→ {label}**")
        else:
            st.sidebar.markdown(f"<span style='color:#888'>{label}</span>", unsafe_allow_html=True)

    # --- Main area ---
    board_col, info_col = st.columns([3, 2])

    with board_col:
        st.markdown(f"**Turn {rec['turn']} — Step {step + 1}/{len(timeline)}**")
        st.markdown(f"### {action_label(rec)}")
        st.image(rec["board_img"], use_container_width=True)

    with info_col:
        st.markdown("### Player State")
        for color in play_order:
            name  = players.get(color, f"p{color}")
            vp    = rec["state"]["player_vp"].get(color, 0)
            cards = rec["state"]["player_resources"].get(color, [])

            settlements = [v for v, cd in rec["state"]["corners"].items()
                           if cd.get("owner") == color and cd.get("buildingType") == 1]
            cities      = [v for v, cd in rec["state"]["corners"].items()
                           if cd.get("owner") == color and cd.get("buildingType") == 2]
            roads       = [e for e, ed in rec["state"]["edges"].items()
                           if ed.get("owner") == color]

            dev_cards     = rec["state"]["player_dev_cards"].get(color, [])
            knights       = rec["state"]["player_knights_played"].get(color, 0)
            road_len      = rec["state"]["player_road_length"].get(color, 0)
            has_army      = rec["state"].get("largest_army_holder") == color
            has_long_road = rec["state"].get("longest_road_holder") == color
            hex_color     = PLAYER_MCOLORS.get(color, "#ccc")
            st.markdown(
                render_player_panel_html(
                    name, vp, cards, dev_cards, settlements, cities, roads, hex_color,
                    knights_played=knights, road_length=road_len,
                    has_largest_army=has_army, has_longest_road=has_long_road,
                ),
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**Balanced Dice Distribution**")
        dice_img = _render_dice_chart(rec["dice_snapshot"], play_order, players)
        st.image(dice_img, use_container_width=True)

        st.markdown("---")
        st.markdown("### Action detail")
        st.json(rec["action"])

        if rec["resources_before"] != rec["resources_after"]:
            RNAMES = {1: "wood", 2: "brick", 3: "sheep", 4: "wheat", 5: "ore"}
            before = [RNAMES.get(r, r) for r in rec["resources_before"]]
            after  = [RNAMES.get(r, r) for r in rec["resources_after"]]
            st.markdown(f"**Resources:** `{before}` → `{after}`")


if __name__ == "__main__":
    main()
