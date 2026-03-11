# CatanBot

A 1v1 Catan tactics trainer built on imitation learning from Colonist.io replays.

---

## Project Goal

Train a policy on real human gameplay captured from Colonist.io, then use it to suggest moves, analyze positions, and eventually play as an opponent. The model operates natively in Colonist's coordinate system — no simulation environment required.

---

## Branch History

| Branch | What was built |
|---|---|
| `main` | Merge point — always a stable snapshot |
| `diceTracker` (merged) | Streamlit balanced-dice tracker app |
| `visualization` | Catan board module: hex grid, topology, board generation, ports, visualizer |
| `balancedDie` | Python port of Colonist.io's balanced dice engine |
| `gameState` | Game state data structures, `new_game()`, all query/helper functions |
| `environment` | Action space (249 actions), state transitions, `CatanEnv` Gym wrapper |
| `imitationLearning` | Replay capture pipeline, parser, Colonist-native training approach |

> Everything from `diceTracker` through `environment` is preserved in `archive/` for potential future use (e.g. self-play, MCTS).

---

## What's in `imitationLearning` (this branch)

### Problem
To train via imitation learning we need `(state, action)` pairs from strong human players. Colonist.io replays contain the full event history of each game, but the API is protected by Cloudflare Turnstile tokens — plain HTTP requests always return 403.

### Solution
Use Playwright with a real Chrome instance and automation-detection suppression to intercept the `data-from-game-id` API response as the browser loads a replay page. This generates a valid Turnstile token automatically.

### Files

| File | Description |
|---|---|
| `colonist/capture.py` | Downloads replay JSON files from Colonist.io |
| `colonist/parse_replay.py` | Parses raw replay JSON → structured `(state, action)` timeline |
| `colonist/colonist_to_catan_mapping.json` | Maps Colonist vertex/edge IDs to CatanEnv IDs (54 vertices, 62/72 edges mapped) |
| `colonist/replays/` | Downloaded raw replay JSONs (gitignored) |
| `docs/` | Strategy notes and reward spec docs |

---

## Setup

```bash
pip install playwright
python -m playwright install chromium
```

### One-time cookie login (required for premium replay access)

Close all Chrome windows, then:

```bash
python colonist/capture.py --login
```

A Chrome window opens with your real profile. Log in to Colonist.io, then press Enter in the terminal. Your session cookies are saved to `colonist/colonist_cookies.json` (gitignored).

> You need a **Colonist.io premium account** to access replays of games you didn't play in.

---

## Usage

### Download replays

```bash
# Single game
python colonist/capture.py 209525825

# Batch
python colonist/capture.py 209525825 213507030 214191014

# Custom output folder
python colonist/capture.py --dir colonist/my_replays 209525825 213507030
```

Saves to `colonist/replays/<game_id>.json` by default.

### Parse a replay

```bash
python colonist/parse_replay.py colonist/replays/209525825.json
```

Prints a turn-by-turn action timeline and saves an `actions.json` with records of the form:

```json
{
  "event_idx": 12,
  "turn": 4,
  "player": 1,
  "player_name": "Dashed",
  "action": { "type": "dice_roll", "dice": [3, 4], "total": 7 },
  "resources_before": [1, 2, 3],
  "resources_after": [1, 2, 3],
  "vp": 3,
  "buildings": { "settlements": ["22", "7"], "cities": [], "roads": ["26", "6"] },
  "robber_tile": 9
}
```

---

## How Capture Works

1. Playwright launches real Chrome (`channel="chrome"`) with `--disable-blink-features=AutomationControlled` and overrides `navigator.webdriver = undefined`
2. Your saved session cookies are injected, authenticating you as a premium user
3. The browser navigates to `colonist.io/replay?gameId=<id>&playerColor=<color>`
4. Cloudflare Turnstile runs normally (it sees a real, non-headless Chrome)
5. The `data-from-game-id` API response is intercepted and saved to disk
6. The browser page closes; repeat for the next game ID

---

## Roadmap

1. ✅ Replay capture pipeline (`capture.py`)
2. ✅ Replay parser — `(state, action)` records (`parse_replay.py`)
3. ⬜ Feature encoder — fixed-size state vector from Colonist game state
4. ⬜ Action encoder — map action types + vertex/edge IDs to action indices
5. ⬜ Behavioral cloning training loop
6. ⬜ Tactics trainer UI — suggest moves during live Colonist.io games
