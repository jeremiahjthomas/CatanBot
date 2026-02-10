# Catan Balanced Dice Visualizer (Manual Dice Tracker)
README.md

Catan Balanced Dice Visualizer (Manual Dice Tracker)

This project is a Streamlit GUI that visualizes a balanced dice / balanced deck system for Catan, reverse-engineered from Colonist’s DiceControllerBalanced logic.

This version is a manual dice tracker:
- You manually input dice rolls (2–12)
- The app shows the true next-roll probability distribution
- Probabilities update based on deck depletion, recent-roll penalties, and player-specific 7 fairness rules

The architecture is intentionally designed so it can later be connected to live Colonist chat ingestion.

--------------------------------------------------
File Overview
--------------------------------------------------

- diceTracker.py — Streamlit app + balanced dice engine (manual roll input)
- README.md — setup and usage instructions (this file)

--------------------------------------------------
Requirements
--------------------------------------------------

- Python 3.9+ (3.10 / 3.11 recommended)
- Anaconda or Miniconda
- Python packages:
  - streamlit
  - pandas
  - altair

--------------------------------------------------
Environment Setup (Anaconda)
--------------------------------------------------

1. Open Anaconda Prompt

On Windows:
Start Menu → Anaconda Prompt

2. Create and activate a virtual environment

conda create -n catanbot python=3.11 -y
conda activate catanbot

3. Install dependencies

pip install streamlit pandas altair

(Optional)
pip freeze > requirements.txt

--------------------------------------------------
Running the App
--------------------------------------------------

Navigate to the project directory and launch Streamlit:

cd C:\Users\madha\CatanBot
conda activate catanbot
streamlit run diceTracker.py

Streamlit will start a local server and print a URL, usually:
http://localhost:8501

Open this link in your browser.

--------------------------------------------------
How to Use
--------------------------------------------------

Game Setup (Sidebar)

1. Enter names for Player 1 and Player 2
2. Click Start / Reset Game

This initializes a fresh balanced-dice engine and clears roll history.

Main Controls

- Current player  
  Chooses whose perspective the adjusted distribution is shown from  
  (important for 7 fairness logic)

- Manual input: rolled total  
  Select a dice total (2–12)

- Apply Roll  
  Applies the roll to the engine and updates all charts

- Reshuffle Deck  
  Refills the deck to the standard 36-card distribution  
  (does NOT clear recent-roll memory or 7 streak state)

- Undo Last  
  Rebuilds engine state by replaying roll history minus the last roll

--------------------------------------------------
What the Visualizations Mean
--------------------------------------------------

Base Distribution (Deck-Only)

Probability is based purely on remaining cards in the deck:

P(next = t) = remaining cards for t / cards left in deck

Adjusted Distribution

Starting from the base distribution, the engine applies:

1. Recent-roll penalty  
   Totals that appeared recently are temporarily down-weighted

2. 7 fairness and streak adjustment (player-specific)  
   - Penalizes or boosts 7s based on imbalance  
   - Accounts for streaks of consecutive 7s by the same player

The distribution is then renormalized so probabilities sum to 1.

--------------------------------------------------
Engine Rules (Important)
--------------------------------------------------

- The deck contains 36 cards with standard 2d6 frequencies:
  2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:5, 9:4, 10:3, 11:2, 12:1

- Pre-draw reshuffle rule  
  If fewer than 13 cards remain, the deck reshuffles before the next roll

- If a manually entered roll has no remaining cards, the engine reshuffles to recover

--------------------------------------------------
Troubleshooting
--------------------------------------------------

streamlit command not found

Make sure the environment is activated:
conda activate catanbot

Verify installation:
streamlit --version

App loads but doesn’t update

- Ensure Apply Roll is clicked
- Refresh the browser
- Check the Streamlit terminal for errors

--------------------------------------------------
Planned Extensions
--------------------------------------------------

- Automatic ingestion of Colonist.io chat dice rolls
- Live per-player statistics
- Comparison with standard (unbalanced) 2d6 dice

--------------------------------------------------
Notes
--------------------------------------------------

This project is for visualization and understanding of balanced dice mechanics, not competitive advantage.

Have fun exploring the math behind the dice.
