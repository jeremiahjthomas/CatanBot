# CatanBot — Project Overview

## End Goal

Build a **1v1 Catan tactics trainer** that watches a live Colonist.io game and
suggests the best move in real time. The bot should play at or above the level
of a strong human player.

The target platform is Colonist.io under its **1v1 competitive ruleset**:

- First to 15 VP wins
- Balanced (deck-based) dice
- Discard threshold: 10+ cards on a 7
- Friendly Robber: can only rob a player with 3+ visible VP
- No player-to-player trading (bank/port only)
- Development cards and ports enabled

---

## How We Got Here — Full History

### Phase 1 — PPO Self-Play (archived)

The original approach was reinforcement learning via Proximal Policy Optimization
(PPO) in a custom Gym environment that fully simulated Catan rules.

**What was built:**

| Component | Description |
|---|---|
| `archive/env/catan_env.py` | Full Gym-compatible 1v1 Catan simulator |
| `archive/env/game_state.py` | Board topology, legal-move generators, VP tracking |
| `archive/env/board.py` | Hex grid, vertex/edge adjacency, port definitions |
| `archive/env/actions.py` | 249-action discrete space (settlements, cities, roads, robber, dev cards, trades) |
| `archive/env/balanced_dice.py` | Reverse-engineered Colonist.io balanced-dice engine |
| `archive/env/policy_net.py` | Actor-critic network with action masking |
| `archive/env/reward_calculator.py` | Potential-based reward shaping |
| `archive/training/train.py` | Vectorized PPO loop, 5M steps, LR annealing, GAE |
| `archive/training/heuristic_players.py` | Three rule-based opponents: `road_builder`, `ows`, `balanced` |
| `archive/training/evaluate.py` | Win-rate evaluation vs. random / heuristic baselines |
| `archive/training/mcts_bot.py` | Flat Monte Carlo + UCB tree search bot |

**Why it was archived:**

Training converged to mediocre play — the simulated environment diverged from
real Colonist.io rules/behavior in subtle ways, and self-play in a sparse-reward
game required enormous compute. The setup phase in particular was hard to learn
from scratch. The agents never reached human-competitive level.

**Key docs produced during this phase:**

- `docs/catan_ppo_reward_spec.md` — reward shaping formula and rationale
- `docs/catan_1v1_initial_placements.md` — opening placement strategy analysis
- `docs/placement_strategy.md` — settlement placement heuristics

---

### Phase 2 — Imitation Learning from Colonist.io Replays (current)

**Pivot rationale:** Instead of learning Catan from scratch via RL, clone the
behavior of real human players from actual game replays. This sidesteps the
reward shaping problem and produces a bot that naturally reflects how good
players actually play.

**What was built:**

#### Data Collection Pipeline

| File | Description |
|---|---|
| `colonist/capture.py` | Playwright + real Chrome replay downloader. Bypasses Cloudflare Turnstile by running a real (non-headless) browser with automation flags suppressed. Intercepts the `data-from-game-id` API response and writes raw JSON to `colonist/replays/`. |
| `colonist/colonist_cookies.json` | Saved session cookies for premium Colonist.io account (gitignored). |
| `colonist/replays/*.json` | Raw replay files downloaded from Colonist.io. Currently ~6 games. |

**Why Playwright with real Chrome (not requests or headless Chromium):**
- The `data-from-game-id` endpoint requires a Cloudflare Turnstile token that only a real browser JS runtime can generate.
- Google OAuth (used by Colonist) blocks Playwright's bundled Chromium as "unsupported browser."
- `--disable-blink-features=AutomationControlled` + `navigator.webdriver = undefined` hides the automation fingerprint.

#### Replay Parsing

| File | Description |
|---|---|
| `colonist/parse_replay.py` | Reconstructs cumulative game state by replaying the event diff stream. Extracts structured `(state, action)` records from `gameLogState` entries. |
| `colonist/colonist_to_catan_mapping.json` | Bidirectional mapping between Colonist.io vertex/edge IDs and the internal coordinate system. Coverage: 54/54 vertices, 62/72 edges (10 missing; 7 appear in real games). |

**Action types extracted by the parser:**

`dice_roll`, `build_settlement`, `build_city`, `build_road`, `buy_dev_card`,
`play_dev_card`, `place_robber`, `steal`, `end_turn`, `receive_resources`,
`discard`, `monopoly`, `trade`, `game_over`

**Output schema per action record:**

```json
{
  "event_idx": int,
  "turn": int,
  "player": int,
  "player_name": str,
  "action": { "type": str, "..." },
  "resources_before": [int],
  "resources_after": [int],
  "vp": int,
  "buildings": {
    "settlements": [vertex_id],
    "cities": [vertex_id],
    "roads": [edge_id]
  },
  "robber_tile": int
}
```

#### Replay Viewer (Debugging / Analysis Tool)

| File | Description |
|---|---|
| `colonist/replay_viewer.py` | Streamlit app. Loads any downloaded replay, renders the board at every action step, shows an action log, action detail panel, per-player resource/VP/dev-card state, and a live Balanced Dice Distribution chart showing the real-time probability of each roll total for each player. |

Features:
- Step-through navigation (per-action and per-turn buttons + slider)
- Board rendered from game state using matplotlib (hexes, settlements, cities, roads, robber)
- Balanced Dice Distribution: per-player bar charts driven by the same engine Colonist uses, updated live as rolls are applied
- Action log with full history; action detail as JSON

---

## Current State

| Area | Status |
|---|---|
| Replay capture | Working |
| Replay parsing | Working (all 14 action types handled) |
| Coordinate mapping | 54/54 vertices, 62/72 edges |
| Replay viewer / debugger | Working |
| Replays collected | ~6 games |
| Feature encoder | Not started |
| Action encoder | Not started |
| Behavioral cloning training loop | Not started |
| Evaluation pipeline | Not started (archived PPO eval exists for reference) |
| Live game advisor UI | Not started |

---

## Remaining Work — What Needs to Happen

### Step 1 — Collect More Data

6 replays is enough to validate the pipeline but far too few to train a
generalizable policy. Behavioral cloning typically needs hundreds to thousands
of games. The capture pipeline is ready; this is a matter of running it.

**Questions to resolve:**
- How many replays do we want? (Recommendation: 500–1000 minimum)
- Filter criteria: only games the target player won? Only games above a
  certain length / VP threshold? All games regardless of outcome?
- Which player perspective to train on (both players in a replay, or only
  the winner, or only the target user)?

---

### Step 2 — Feature Encoder

Convert a parsed state record into a fixed-size float32 vector for neural
network input.

**Candidate dimensions:**

| Feature group | Encoding | Approx dims |
|---|---|---|
| Tile layout (resource type per hex) | One-hot × 19 hexes | 19 × 6 = 114 |
| Tile dice numbers | One-hot or normalized scalar × 19 | 19 × 11 = 209 |
| Vertex ownership (mine / opponent / empty) | One-hot × 54 | 54 × 3 = 162 |
| Edge ownership (mine / opponent / empty) | One-hot × 72 | 72 × 3 = 216 |
| Robber tile | One-hot | 19 |
| Current player resources (by type) | Count vector | 5 |
| Opponent resources (total, hidden) | Scalar | 1 |
| Current player VP | Scalar | 1 |
| Opponent VP | Scalar | 1 |
| Dev cards in hand (by type) | Count vector | 5 |
| Knights played (self/opponent) | Scalar × 2 | 2 |
| Longest road (holder flag) | Binary × 2 | 2 |
| Largest army (holder flag) | Binary × 2 | 2 |
| Deck composition (remaining cards per roll total) | 11 scalars | 11 |
| Whose turn | Binary | 1 |

Approximate total: **~750 dims** (can be reduced with ablations)

**Open design questions:**
- Should the board layout (tiles) be included at all, or is it fixed per game
  and should be provided separately?
- Should opponent resources be fully hidden (only total count) or partially
  inferred from observation history?
- Should we use raw counts or normalize everything to [0, 1]?

---

### Step 3 — Action Encoder

Map each action in the replay to a flat integer index for classification.

**Candidate action space:**

| Action | Indices | Count |
|---|---|---|
| Build settlement at vertex v | 0–53 | 54 |
| Build city at vertex v | 54–107 | 54 |
| Build road at edge e | 108–179 | 72 |
| Buy dev card | 180 | 1 |
| Play knight | 181 | 1 |
| Play road building | 182 | 1 |
| Play year of plenty (each resource combo) | 183–? | ~15 |
| Play monopoly (each resource) | ~198–202 | 5 |
| Bank trade (give/receive combos) | ~203–? | ~25 |
| Port trade (give/receive combos) | ~228–? | ~30 |
| End turn | last | 1 |

Approximate total: **~260 actions** (close to the archived 249-action space)

**Open design question:**
- Do we train a single model for all phases (setup + main game), or separate
  models for setup placement vs. main-game actions?
- Setup (initial placement) follows different rules and has no resource cost —
  it may be easier to handle with a separate head or a simple heuristic.

---

### Step 4 — Build the Training Dataset

Run all downloaded replays through the feature encoder + action encoder to
produce `(state_vec, action_idx)` training pairs.

- Filter: only include records where `action.type` is a **decision** action
  (exclude `dice_roll`, `receive_resources`, `end_turn` unless we want to
  predict those too)
- Split: 80% train / 10% val / 10% test (by game, not by record, to avoid
  leaking game context across splits)

---

### Step 5 — Behavioral Cloning Loop

Standard supervised learning: cross-entropy loss between predicted action
distribution and the human's actual action.

- Architecture: MLP or small transformer over the state vector
- The archived `policy_net.py` (actor-critic) can be adapted — remove the
  value head, keep the action head
- Action masking: illegal actions should be masked to zero probability before
  computing loss (same masking used in archived PPO code)
- Training: Adam, ~100 epochs, early stopping on validation loss

---

### Step 6 — Evaluation

**Offline metrics (no game needed):**
- Top-1 accuracy: did the model predict exactly what the human did?
- Top-3 / Top-5 accuracy: was the human's move in the model's top-3/5?

**Online metrics (requires playing full games):**
- Win rate vs. archived heuristic opponents (`road_builder`, `ows`, `balanced`)
- Win rate vs. random agent
- Optionally: win rate vs. the archived PPO checkpoint

**Complication:** The behavioral cloning policy operates in Colonist coordinate
space, but the archived game simulator (`CatanEnv`) uses its own internal
coordinate system. The `colonist_to_catan_mapping.json` file bridges these, but
integration work is needed to evaluate the BC policy in the simulator.

---

### Step 7 — Live Game Advisor (End Goal)

A Streamlit or browser overlay that:
1. Watches a live Colonist.io game via the same Playwright capture mechanism
2. Encodes current game state in real time
3. Runs the trained model to get top-3 suggested moves
4. Displays them as an overlay with probabilities

This is the original end goal. Steps 1–6 are prerequisites.

---

## Finalized Roadmap

### Planning Decisions (Resolved)

| Question | Decision |
|---|---|
| Data volume | 500–1000 replays minimum. Both players per replay, wins and losses, with metadata. |
| Training target | All decision actions from both players. Slightly upweight winner actions. |
| Setup phase | Separate model or separate policy head. Do not merge blindly into main action space. |
| Evaluation environment | Replay-based evaluation first. Archived simulator only as secondary integration check. |
| Architecture | Start with a masked MLP baseline. Do not start with a GNN. |
| Scope of advisor | Post-game review tool first (in the replay viewer), then promote to live advisory. |

**Overall stance:** The most important things right now are more data, correct
information constraints, clean action encoding, and tight replay-based evaluation.
That combination will matter far more than model architecture.

---

## Phase A — Clean Data Foundation

The goal of this phase is not to train. The goal is to make replay records,
mappings, and state reconstruction so reliable that when the model fails later,
you know it is a modeling problem and not a data bug.

**Outputs of Phase A:**
- Larger replay corpus (500+ games)
- Full or near-full coordinate mapping coverage
- Stable parsed replay schema
- Observed-information state representation (not omniscient)
- Validation scripts that catch bad replays automatically

---

### Phase A.1 — Replay Corpus Manifest

Before collecting hundreds of replays, standardize storage and metadata.

**Create:** `colonist/replays/manifest.jsonl` — one JSON record per replay.

**Fields per entry:**

| Field | Type | Description |
|---|---|---|
| `game_id` | str | Colonist game ID |
| `file_path` | str | Path to raw replay JSON |
| `download_timestamp` | str | ISO timestamp |
| `parser_version` | str | Version of parse_replay.py used |
| `colonist_ruleset` | str | e.g. `"1v1_competitive"` |
| `players` | list | `[{color, name}]` |
| `winner` | int | Winning player color |
| `final_vp_p0` | int | |
| `final_vp_p1` | int | |
| `turn_count` | int | |
| `action_count` | int | |
| `parse_success` | bool | Did parser complete without error? |
| `parse_warnings` | list | Non-fatal issues found |
| `mapping_coverage_ok` | bool | All edges/vertices mapped? |
| `has_unknown_events` | bool | Any unrecognized log types? |
| `has_unknown_edges` | bool | Any edge IDs not in mapping? |
| `has_unknown_vertices` | bool | Any vertex IDs not in mapping? |
| `game_length_bucket` | str | `"short"` / `"medium"` / `"long"` |
| `notes` | str | Free-text |

**Why this matters:** Once you have 500+ games you need to answer questions
like "which replays parsed cleanly?", "which used an old parser version?",
"which contain unmapped roads?", "which are too short to use?" Without a
manifest this becomes painful fast.

---

### Phase A.2 — Fix Edge Mapping Coverage

Currently 10 of 72 edges are unmapped, 7 of which appear in real games.
Complete the mapping before training so no road-build actions are silently
dropped.

---

### Phase A.3 — Observed-Information State

The parser currently reconstructs a fully omniscient state (both players'
exact resources, all dev cards). A real model should only see what its player
can observe:

- **Own resources:** exact counts by type
- **Opponent resources:** total card count only (hidden)
- **Own dev cards:** exact (type known)
- **Opponent dev cards:** count only (type hidden until played)
- **Dice deck:** remaining counts per total (public information in Colonist)
- **Board:** fully public (all buildings and roads visible)

This distinction matters for training: the model should learn to play under
uncertainty, not assume it can see the opponent's hand.

---

## Phase B — Offline Dataset Builder

Create a script that runs all parsed replays through the feature encoder and
action encoder and writes out a flat dataset.

**Each row in the dataset:**

| Field | Description |
|---|---|
| `state_vec` | float32 vector, observed-information state |
| `legal_action_mask` | binary vector over full action space |
| `action_idx` | index of the action the player took |
| `phase` | `"setup"` or `"main"` |
| `action_type` | human-readable type string for debugging |
| `game_id` | source game |
| `player_id` | which player in the game |
| `winner_flag` | did this player win? |

**Also store a human-readable sidecar (JSON or CSV) with:**
- Decoded action string
- Decoded legal moves
- State snapshot (for spot-checking individual rows)

**Train/val/test split:** 80/10/10 by game, not by record, to prevent
leaking game context across splits.

---

## Phase C — Train the Baseline

Train two models:

1. **Setup model** — predicts initial settlement and road placements
2. **Main-game masked MLP** — predicts all decision actions during the main game

**Metrics to track:**
- Top-1 accuracy overall
- Top-3 and Top-5 accuracy overall
- Per-action-type accuracy (build settlement, build road, buy dev card, etc.)
- Setup placement accuracy separately

Do not report a single aggregate accuracy and call it done. Per-action-type
breakdown will expose where the model is actually weak.

---

## Phase D — Model in the Replay Viewer

This is the highest-value next product milestone. Before deploying anything
live, add model predictions to the existing replay viewer.

**For each action step in the viewer, show:**
- The actual action the human took
- The model's top-3 predicted actions with probabilities
- Whether the human's move was in the model's top-3
- A running top-3 hit rate for the game

This tells you far more than a loss curve. If the model's top-3 consistently
includes sensible moves but misses the exact human choice, that is a very
different problem than if it is predicting illegal or nonsensical actions.

---

## Phase E — Improve Policy Quality

Only after the Phase C baseline is working and Phase D review is showing
sensible predictions:

- **Winner upweighting** — increase loss weight on actions from the winning player
- **Phase-specific heads** — separate network heads for setup / early game / late game
- **Action-family factorization** — predict action type first, then target vertex/edge
- **Inferred hidden-state features** — model opponent's likely hand from observation history
- **Ranking / contrastive loss** — penalize preferring illegal or dominated moves
- **GNN / graph-aware encoder** — replace flat MLP with a board-topology-aware architecture

---

## Phase F — Live Game Advisor

Only once replay review predictions are consistently sensible:

1. Extend the Playwright capture mechanism to watch a live game in real time
2. Encode current observed state on each action prompt
3. Run the trained model to get top-3 suggestions with probabilities
4. Display as an overlay alongside the Colonist.io browser window

Build the post-game review tool in Phase D first. Phase F is a direct
promotion of that same UI to a live context.
