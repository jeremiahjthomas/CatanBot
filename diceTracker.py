
import streamlit as st
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd
import altair as alt

TOTALS = list(range(2, 13))

@dataclass
class SevenStreak:
    player: Optional[str] = None
    count: int = 0


class BalancedDiceEngine:
    """
    Reverse-engineered from the provided TypeScript controller (DiceControllerBalanced).

    We model the deck as counts of remaining outcome 'cards' per total (2..12),
    which is sufficient for computing the next-roll distribution.
    """

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

        # Constants (match TS defaults)
        self.minimum_cards_before_reshuffling = minimum_cards_before_reshuffling
        self.prob_reduction_recent = probability_reduction_for_recently_rolled
        self.prob_reduction_seven_streaks = probability_reduction_for_seven_streaks
        self.maximum_recent_roll_memory = maximum_recent_roll_memory

        # Deck state
        self.deck_counts: Dict[int, int] = {t: 0 for t in TOTALS}
        self.cards_left: int = 0

        # Recent-roll memory (windowed)
        self.recent_rolls: List[int] = []
        self.recently_rolled_count: Dict[int, int] = {t: 0 for t in TOTALS}

        # 7 fairness / streak state
        self.seven_streak = SevenStreak(player=None, count=0)
        self.total_sevens_by_player: Dict[str, int] = {}  # lazy init like TS

        self.reshuffle()

    @staticmethod
    def standard_counts() -> Dict[int, int]:
        # 2d6 totals distribution across 36 outcomes
        return {2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 5, 9: 4, 10: 3, 11: 2, 12: 1}

    def reshuffle(self) -> None:
        """
        Refill the deck to the standard 36-card multiset.
        Mirrors TS: reshuffle refills dicePairs + sets cardsLeftInDeck=36.
        It does NOT reset recent rolls or sevens fairness state.
        """
        std = self.standard_counts()
        for t in TOTALS:
            self.deck_counts[t] = std[t]
        self.cards_left = 36

    # ----- internal helpers mirroring TS -----
    def _init_total_sevens(self, player: str) -> None:
        if player not in self.total_sevens_by_player:
            self.total_sevens_by_player[player] = 0

    def _update_recent_window(self) -> None:
        # Mirrors updateRecentlyRolled(): keep only last `maximum_recent_roll_memory` totals
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
        # TS: isStreakForOrAgainstPlayer = streakPlayer == player ? -1 : +1
        if self.seven_streak.player is None:
            return 0.0
        sign = -1 if self.seven_streak.player == player else 1
        return self.prob_reduction_seven_streaks * self.seven_streak.count * sign

    def _get_seven_imbalance_adjustment(self, player: str) -> float:
        total_sevens = self._get_total_sevens_rolled()
        initialized_players = len(self.total_sevens_by_player)
        # TS: if totalSevens < map.size return 1
        if initialized_players == 0:
            return 1.0
        if total_sevens < initialized_players:
            return 1.0
        if total_sevens == 0:
            return 1.0

        player_sevens = self.total_sevens_by_player.get(player, 0)
        percentage_of_total = player_sevens / total_sevens
        ideal_percentage = 1.0 / initialized_players
        # TS: 1 + ((ideal - actual) / ideal)
        return 1.0 + ((ideal_percentage - percentage_of_total) / ideal_percentage)

    def _seven_probability_multiplier_for_player(self, player: str) -> float:
        # TS: if numberOfPlayers < 2 return
        if self.number_of_players < 2:
            return 1.0

        self._init_total_sevens(player)
        streak_adj = self._get_streak_adjustment_constant(player)
        imbalance_adj = self._get_seven_imbalance_adjustment(player)

        # TS: sevenProbabilityAdjustment = 1 * playerSevensAdjustmentPercentage + streakAdjustmentPercentage
        seven_adj = 1.0 * imbalance_adj + streak_adj

        # clamp [0, 2]
        seven_adj = max(0.0, min(2.0, seven_adj))
        return seven_adj

    # ----- public API -----
    def base_distribution(self) -> Dict[int, float]:
        if self.cards_left <= 0:
            self.reshuffle()
        return {t: self.deck_counts[t] / self.cards_left for t in TOTALS}

    def adjusted_distribution(self, player: str) -> Dict[int, float]:
        """
        Next-roll distribution after:
        - base deck probabilities (remaining cards / cards_left)
        - recent-roll penalty per total
        - seven fairness/streak adjustment (applied to 7 only for this player)
        Then renormalized for display.
        """
        base = self.base_distribution()

        # recent-roll penalty
        adjusted = {}
        for t in TOTALS:
            reduction = self.recently_rolled_count[t] * self.prob_reduction_recent
            multiplier = 1.0 - reduction
            if multiplier < 0:
                multiplier = 0.0
            adjusted[t] = base[t] * multiplier

        # seven adjustment
        adjusted[7] *= self._seven_probability_multiplier_for_player(player)

        # renormalize
        s = sum(adjusted.values())
        if s <= 0:
            # fallback to base renormalized
            sb = sum(base.values())
            return {t: (base[t] / sb if sb > 0 else 0.0) for t in TOTALS}
        return {t: adjusted[t] / s for t in TOTALS}

    def apply_roll(self, player: str, total: int) -> None:
        """
        Update state as if `player` rolled `total`.

        Note: TS reshuffles before draw if cardsLeftInDeck < minimumCardsBeforeReshuffling.
        """
        if total not in TOTALS:
            raise ValueError("total must be 2..12")

        # Pre-draw reshuffle rule
        if self.cards_left < self.minimum_cards_before_reshuffling:
            self.reshuffle()

        # Remove one card from that total
        if self.deck_counts[total] <= 0:
            # manual input could force impossible states; recover by reshuffle
            self.reshuffle()
            if self.deck_counts[total] <= 0:
                raise RuntimeError(f"No cards available for total={total} even after reshuffle.")

        self.deck_counts[total] -= 1
        self.cards_left -= 1

        # Recent-roll memory updates
        self.recent_rolls.append(total)
        self.recently_rolled_count[total] += 1
        self._update_recent_window()

        # 7 fairness/streak tracking
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


# ---------------- Streamlit App ----------------

st.set_page_config(page_title="Catan Balanced Dice Visualizer", layout="wide")

st.title("Catan Balanced Dice Visualizer (Balanced Deck + Adjustments)")
st.caption("Manual input now; designed to plug into Colonist chat ingestion later.")

with st.sidebar:
    st.header("Game Setup")
    n_players = 2
    st.write("This MVP is fixed to **2 players (1v1)**.")
    default_names = ["Player 1", "Player 2"]

    # Let user customize player names
    st.subheader("Player names")
    name_p1 = st.text_input("Name for Player 1", value=default_names[0], key="name_0")
    name_p2 = st.text_input("Name for Player 2", value=default_names[1], key="name_1")
    names = [name_p1, name_p2]


    if st.button("Start / Reset Game", type="primary"):

        st.session_state.engine = BalancedDiceEngine(
            players=names,
            minimum_cards_before_reshuffling=13,
            probability_reduction_for_recently_rolled=0.34,
            probability_reduction_for_seven_streaks=0.40,
            maximum_recent_roll_memory=5,
        )
        st.session_state.history = []  # list of (player, total)
        st.success("Game reset.")

# Initialize if missing
if "engine" not in st.session_state:
    st.session_state.engine = BalancedDiceEngine(players=["Player 1","Player 2"])
if "history" not in st.session_state:
    st.session_state.history = []

engine: BalancedDiceEngine = st.session_state.engine

# Main controls row
c1, c2, c3, c4 = st.columns([2, 2, 2, 2])

with c1:
    current_player = st.selectbox("Current player (distribution depends on this)", engine.players, index=0)

with c2:
    rolled_total = st.selectbox("Manual input: rolled total", TOTALS, index=5)  # default 7

with c3:
    if st.button("Apply Roll", use_container_width=True):
        try:
            engine.apply_roll(current_player, int(rolled_total))
            st.session_state.history.append((current_player, int(rolled_total)))
            st.toast(f"Applied roll: {current_player} rolled {rolled_total}", icon="🎲")
        except Exception as e:
            st.error(str(e))

with c4:
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reshuffle Deck", use_container_width=True):
            engine.reshuffle()
            st.toast("Deck reshuffled (memory not cleared).", icon="🃏")
    with col_b:
        if st.button("Undo Last", use_container_width=True):
            # easiest undo: rebuild from scratch using history minus last
            if st.session_state.history:
                st.session_state.history.pop()
                # Recreate engine with same params but fresh state, then replay history
                engine2 = BalancedDiceEngine(
                    players=engine.players,
                    minimum_cards_before_reshuffling=engine.minimum_cards_before_reshuffling,
                    probability_reduction_for_recently_rolled=engine.prob_reduction_recent,
                    probability_reduction_for_seven_streaks=engine.prob_reduction_seven_streaks,
                    maximum_recent_roll_memory=engine.maximum_recent_roll_memory,
                )
                for p, t in st.session_state.history:
                    engine2.apply_roll(p, t)
                st.session_state.engine = engine2
                engine = engine2
                st.toast("Undid last roll.", icon="↩️")
            else:
                st.info("No rolls to undo.")

st.divider()

# Compute distributions
base = engine.base_distribution()
adj = engine.adjusted_distribution(current_player)
MAX_Y = max(1/6, max(adj.values()), max(base.values()))

# Build a display table (no pandas required)
rows = []
for t in TOTALS:
    rows.append({
        "Total": t,
        "Remaining Cards": engine.deck_counts[t],
        "Base P(next)": base[t],
        "Adjusted P(next)": adj[t],
        "Recent Count (window)": engine.recently_rolled_count[t],
    })

left, right = st.columns([1.3, 1])

with left:
    st.subheader("Next-roll distribution")
    st.write("**Base** comes from the remaining deck. **Adjusted** applies recent-roll penalties + the player-specific 7 fairness/streak multiplier, then renormalizes.")

    # Streamlit bar charts work best with a dict/list structure
    # We'll use two charts for clarity
    st.write("**Adjusted distribution (for selected player)**")
    df_adj = pd.DataFrame({"Total": TOTALS, "Probability": [adj[t] for t in TOTALS]})
    chart_adj = (
        alt.Chart(df_adj)
        .mark_bar()
        .encode(
            x=alt.X("Total:O", sort=TOTALS, axis=alt.Axis(title="Total (2d6)")),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, MAX_Y]), axis=alt.Axis(title="Probability")),
            tooltip=[alt.Tooltip("Total:O"), alt.Tooltip("Probability:Q", format=".6f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart_adj, use_container_width=True)

    st.write("**Base distribution (deck-only)**")
    df_base = pd.DataFrame({"Total": TOTALS, "Probability": [base[t] for t in TOTALS]})
    chart_base = (
        alt.Chart(df_base)
        .mark_bar()
        .encode(
            x=alt.X("Total:O", sort=TOTALS, axis=alt.Axis(title="Total (2d6)")),
            y=alt.Y("Probability:Q", scale=alt.Scale(domain=[0, MAX_Y]), axis=alt.Axis(title="Probability")),
            tooltip=[alt.Tooltip("Total:O"), alt.Tooltip("Probability:Q", format=".6f")],
        )
        .properties(height=320)
    )
    st.altair_chart(chart_base, use_container_width=True)

    st.subheader("Table view")
    df_table = pd.DataFrame(rows)
    # Ensure totals are ordered 2..12
    df_table["Total"] = pd.Categorical(df_table["Total"], categories=TOTALS, ordered=True)
    df_table = df_table.sort_values("Total")
    df_table = df_table.set_index("Total")
    st.table(df_table)

with right:
    st.subheader("State / Debug")

    snap = engine.snapshot()

    st.metric("Cards left in deck", snap["cards_left"])
    st.write("**Recent rolls** (most recent at end):", snap["recent_rolls"])

    st.write("**7 streak**")
    st.write(f"- streak player: `{snap['seven_streak_player']}`")
    st.write(f"- streak count: `{snap['seven_streak_count']}`")

    st.write("**Total 7s by player**")
    for p in engine.players:
        st.write(f"- {p}: {snap['total_sevens_by_player'].get(p, 0)}")

    # show the 7 multiplier for current player
    seven_mult = engine._seven_probability_multiplier_for_player(current_player)
    st.write("**Current 7 multiplier for selected player**")
    st.code(f"{seven_mult:.4f}", language="text")

    st.subheader("Roll history")
    if st.session_state.history:
        st.write("Most recent first:")
        for p, t in reversed(st.session_state.history[-50:]):
            st.write(f"- {p} rolled **{t}**")
    else:
        st.write("No rolls yet.")
