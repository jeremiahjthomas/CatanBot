# streamlit_app.py
import time
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import streamlit as st
import pandas as pd
import altair as alt

TOTALS = list(range(2, 13))
Y_MAX = 1 / 6

STATE_PATH = Path(__file__).resolve().parent / "roll_state.json"


def safe_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
        return
    try:
        from streamlit.runtime.scriptrunner import RerunException, RerunData
        raise RerunException(RerunData())
    except Exception:
        return


def load_roll_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception as e:
            return {"last_id": 0, "events": [], "error": str(e)}
    return {"last_id": 0, "events": []}


@dataclass
class SevenStreak:
    player: Optional[str] = None
    count: int = 0


class BalancedDiceEngine:
    def __init__(
        self,
        players: List[str],
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

        self.deck_counts: Dict[int, int] = {t: 0 for t in TOTALS}
        self.cards_left: int = 0

        self.recent_rolls: List[int] = []
        self.recently_rolled_count: Dict[int, int] = {t: 0 for t in TOTALS}

        self.seven_streak = SevenStreak(player=None, count=0)
        self.total_sevens_by_player: Dict[str, int] = {}

        self.reshuffle()

    @staticmethod
    def standard_counts() -> Dict[int, int]:
        return {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

    def reshuffle(self) -> None:
        std = self.standard_counts()
        for t in TOTALS:
            self.deck_counts[t] = std[t]
        self.cards_left = 36

    def _init_total_sevens(self, player: str) -> None:
        if player not in self.total_sevens_by_player:
            self.total_sevens_by_player[player] = 0

    def _update_recent_window(self) -> None:
        if len(self.recent_rolls) <= self.maximum_recent_roll_memory:
            return
        oldest = self.recent_rolls.pop(0)
        self.recently_rolled_count[oldest] = max(0, self.recently_rolled_count[oldest] - 1)

    def _update_seven_rolls(self, player: str) -> None:
        self._init_total_sevens(player)
        self.total_sevens_by_player[player] += 1
        if self.seven_streak.player == player:
            self.seven_streak.count += 1
        else:
            self.seven_streak.player = player
            self.seven_streak.count = 1

    def _get_total_sevens_rolled(self) -> int:
        return sum(self.total_sevens_by_player.values())

    def _get_streak_adjustment_constant(self, player: str) -> float:
        if self.seven_streak.player is None:
            return 0.0
        sign = -1 if self.seven_streak.player == player else 1
        return self.prob_reduction_seven_streaks * self.seven_streak.count * sign

    def _get_seven_imbalance_adjustment(self, player: str) -> float:
        total_sevens = self._get_total_sevens_rolled()
        initialized_players = len(self.total_sevens_by_player)
        if initialized_players == 0:
            return 1.0
        if total_sevens < initialized_players:
            return 1.0
        if total_sevens == 0:
            return 1.0
        player_sevens = self.total_sevens_by_player.get(player, 0)
        percentage_of_total = player_sevens / total_sevens
        ideal_percentage = 1.0 / initialized_players
        return 1.0 + ((ideal_percentage - percentage_of_total) / ideal_percentage)

    def seven_multiplier_for_player(self, player: str) -> float:
        if self.number_of_players < 2:
            return 1.0
        self._init_total_sevens(player)
        streak_adj = self._get_streak_adjustment_constant(player)
        imbalance_adj = self._get_seven_imbalance_adjustment(player)
        seven_adj = 1.0 * imbalance_adj + streak_adj
        return max(0.0, min(2.0, seven_adj))

    def base_distribution(self) -> Dict[int, float]:
        if self.cards_left <= 0:
            self.reshuffle()
        return {t: self.deck_counts[t] / self.cards_left for t in TOTALS}

    def adjusted_distribution(self, player: str) -> Dict[int, float]:
        base = self.base_distribution()
        adjusted = {}
        for t in TOTALS:
            reduction = self.recently_rolled_count[t] * self.prob_reduction_recent
            multiplier = max(0.0, 1.0 - reduction)
            adjusted[t] = base[t] * multiplier
        adjusted[7] *= self.seven_multiplier_for_player(player)
        s = sum(adjusted.values())
        if s <= 0:
            sb = sum(base.values())
            return {t: (base[t] / sb if sb > 0 else 0.0) for t in TOTALS}
        return {t: adjusted[t] / s for t in TOTALS}

    def apply_roll(self, player: str, total: int) -> None:
        if total not in TOTALS:
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

    def snapshot(self) -> Dict[str, Any]:
        return {
            "cards_left": self.cards_left,
            "recent_rolls": list(self.recent_rolls),
            "recently_rolled_count": dict(self.recently_rolled_count),
            "seven_streak_player": self.seven_streak.player,
            "seven_streak_count": self.seven_streak.count,
            "total_sevens_by_player": dict(self.total_sevens_by_player),
        }


# ---------------- UI ----------------
st.set_page_config(page_title="Catan Balanced Dice (Live)", layout="wide")
st.title("Catan Balanced Dice Visualizer (Live from Colonist chat)")

p1 = st.text_input("Player 1 name", value="Gimpel5777")
p2 = st.text_input("Player 2 name", value="Endermantexa")
players = [p1, p2]

if "engine" not in st.session_state or st.session_state.get("players") != players:
    st.session_state.players = players
    st.session_state.engine = BalancedDiceEngine(players=players)
    st.session_state.last_applied_event_id = 0
    st.session_state.applied_history = []

engine: BalancedDiceEngine = st.session_state.engine

colA, colB, colC = st.columns([2, 2, 2])
with colA:
    perspective = st.selectbox("Distribution perspective", players, index=0)
with colB:
    if st.button("Reset engine state"):
        st.session_state.engine = BalancedDiceEngine(players=players)
        st.session_state.last_applied_event_id = 0
        st.session_state.applied_history = []
        engine = st.session_state.engine
with colC:
    auto_refresh = st.toggle("Auto-refresh (reads roll_state.json)", value=True)

# Always read the file on every run
state = load_roll_state()
file_last_id = int(state.get("last_id", 0))
events = state.get("events", [])

if "last_applied_event_id" not in st.session_state:
    st.session_state.last_applied_event_id = 0

# Apply new events
new_events = [e for e in events if int(e.get("id", 0)) > st.session_state.last_applied_event_id]
for e in new_events:
    if e.get("type") != "dice_roll":
        continue
    player = e.get("player", "")
    total = int(e.get("total", 0))
    if player not in players:
        continue
    engine.apply_roll(player, total)
    st.session_state.applied_history.append((int(e.get("id", 0)), player, total))
    st.session_state.last_applied_event_id = max(st.session_state.last_applied_event_id, int(e.get("id", 0)))

# Debug panel
with st.expander("Debug", expanded=True):
    st.write("STATE_PATH:", str(STATE_PATH))
    st.write("File exists:", STATE_PATH.exists())
    st.write("File last_id:", file_last_id)
    st.write("Events in file:", len(events))
    st.write("Last applied:", st.session_state.last_applied_event_id)
    if "error" in state:
        st.error(state["error"])

st.divider()

base = engine.base_distribution()
adj = engine.adjusted_distribution(perspective)

df_adj = pd.DataFrame({"Total": TOTALS, "Probability": [adj[t] for t in TOTALS]})
df_base = pd.DataFrame({"Total": TOTALS, "Probability": [base[t] for t in TOTALS]})

chart_adj = (
    alt.Chart(df_adj)
    .mark_bar()
    .encode(
        x=alt.X("Total:O", sort=TOTALS, axis=alt.Axis(title="Total (2d6)")),
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, Y_MAX]), axis=alt.Axis(title="Probability")),
        tooltip=[alt.Tooltip("Total:O"), alt.Tooltip("Probability:Q", format=".6f")],
    )
    .properties(height=320, title=f"Adjusted distribution (for {perspective})")
    .interactive(False)
)

chart_base = (
    alt.Chart(df_base)
    .mark_bar()
    .encode(
        x=alt.X("Total:O", sort=TOTALS, axis=alt.Axis(title="Total (2d6)")),
        y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, Y_MAX]), axis=alt.Axis(title="Probability")),
        tooltip=[alt.Tooltip("Total:O"), alt.Tooltip("Probability:Q", format=".6f")],
    )
    .properties(height=320, title="Base distribution (deck-only)")
    .interactive(False)
)

left, right = st.columns([1.5, 1])
with left:
    st.subheader("Next-roll distributions (fixed y-axis 0..1/6)")
    st.altair_chart(chart_adj, use_container_width=True)
    st.altair_chart(chart_base, use_container_width=True)

    rows = []
    for t in TOTALS:
        rows.append({
            "Total": t,
            "Remaining Cards": engine.deck_counts[t],
            "Base P(next)": base[t],
            "Adjusted P(next)": adj[t],
            "Recent Count": engine.recently_rolled_count[t],
        })
    st.subheader("Table (2..12)")
    st.table(pd.DataFrame(rows).set_index("Total"))

with right:
    st.subheader("Applied events")
    st.write(f"File last_id: **{file_last_id}**")
    st.write(f"Last applied: **{st.session_state.last_applied_event_id}**")

    hist = st.session_state.applied_history
    if not hist:
        st.write("No events applied yet.")
    else:
        for (eid, pl, tot) in reversed(hist[-40:]):
            st.write(f"- #{eid}: **{pl}** rolled **{tot}**")

    st.subheader("Engine debug")
    snap = engine.snapshot()
    st.write("Cards left:", snap["cards_left"])
    st.write("Recent rolls:", snap["recent_rolls"])
    st.write("7 streak:", snap["seven_streak_player"], "x", snap["seven_streak_count"])
    st.write("Total 7s:", snap["total_sevens_by_player"])

# ✅ Auto-refresh logic: only rerun if file is ahead, otherwise sleep then rerun
if auto_refresh:
    # if new ids exist, rerun immediately; else poll after delay
    if file_last_id > st.session_state.last_applied_event_id:
        safe_rerun()
    time.sleep(0.75)
    safe_rerun()
