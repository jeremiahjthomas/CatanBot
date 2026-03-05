# CatanBot

A 1v1 Catan RL bot and tactics trainer, designed around the Colonist.io ruleset.

---

## Project Goal

Build an AI that plays Catan well enough to beat an average human, and a tactics trainer that explains why a move is good or bad. The final product runs as a Streamlit app that can analyze live Colonist.io games.

---

## Branch History

| Branch | What was built |
|---|---|
| `main` | Merge point — always a stable snapshot |
| `diceTracker` (merged) | Manual balanced-dice tracker Streamlit app |
| `visualization` | Catan board module: hex grid, topology, board generation, ports, visualizer |
| `balancedDie` | Python port of Colonist.io's balanced dice engine |
| `gameState` | Game state data structures, new_game(), all query/helper functions |
| `botBranch` | *(planned)* Actions, RL environment, agent |

---

## What Exists Now

### Balanced Dice Tracker (`diceTracker.py`)
Streamlit GUI that visualises the Colonist.io balanced-dice engine.
- Manual roll input (2–12)
- Shows true next-roll probability distribution
- Recent-roll penalty, 7 fairness, and streak adjustment

Run it:
```
conda activate catanbot
streamlit run diceTracker.py
```

### Board Module (`env/board.py`)
Full Catan board representation.
- 19 hexes in axial coordinates, 54 vertices, 72 edges
- Topology pre-computed at import (adjacency, vertex/edge lookup tables)
- Board generation with Colonist.io constraints (no adjacent 6&8, 2&12, or same tokens)
- 9 ports, randomised type per game, fixed coastal positions

```python
from env.board import generate_board
board = generate_board(seed=42)
```

### Board Visualiser (`visualize_board.py`)
matplotlib visualisation of any generated board.
```
python visualize_board.py          # random board
python visualize_board.py 42       # seeded
python visualize_board.py 42 save  # save to board_viz.png
```

---

## Environment Setup

```
conda create -n catanbot python=3.11 -y
conda activate catanbot
pip install streamlit pandas altair matplotlib
```

---

## Testing

```
python test_board.py   # 24 board topology & generation tests
```

---

## Roadmap (see SPEC.md on each branch for detail)

1. Board module — done (`visualization` branch)
2. Balanced dice engine (`env/balanced_dice.py`)
3. Game state + action encoding (`env/game_state.py`, `env/actions.py`)
4. Gym environment (`env/catan_env.py`)
5. PPO + LSTM agent with self-play
6. Tactics trainer + explanation module
