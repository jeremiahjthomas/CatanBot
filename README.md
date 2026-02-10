# CatanBot
Catan Balanced Dice Visualizer (Manual Dice Tracker)

This project is a Streamlit GUI that visualizes the “balanced dice” / “balanced deck” system for Catan (reverse-engineered from the TypeScript DiceControllerBalanced logic).

For now, it’s a manual tracker:

You choose the current player

You manually enter the rolled total (2–12)

The app updates the true next-roll distribution based on:

remaining “deck” cards (2d6 totals across 36 outcomes)

recent-roll penalties (windowed memory)

7 fairness / streak adjustment (player-specific)

Later, this can be extended to read Colonist.io chat automatically.

Files

diceTracker.py — Streamlit app + balanced dice engine (manual roll input)

Requirements

Python 3.9+ recommended (3.10/3.11 works great)

Anaconda / Miniconda

Packages: streamlit, pandas, altair

Setup (Anaconda)
1) Open Anaconda Prompt

On Windows: Start Menu → Anaconda Prompt

2) Create a new conda environment (recommended)
conda create -n catanbot python=3.11 -y
conda activate catanbot

3) Install dependencies
pip install streamlit pandas altair


(Optional) Save dependencies:

pip freeze > requirements.txt

Run the app

In Anaconda Prompt:

cd C:\Users\madha\CatanBot
conda activate catanbot
streamlit run diceTracker.py


Streamlit will print a URL (usually):

http://localhost:8501

Open that link in your browser.

How to use
Sidebar: Game Setup

Enter Player 1 and Player 2 names

Click Start / Reset Game

This resets the engine and clears roll history.

Main controls (top row)

Current player: selects whose perspective the adjusted distribution uses
(this matters because the 7 adjustment is player-specific)

Manual input: rolled total: choose 2–12

Apply Roll: updates the engine state and refreshes charts + table

Reshuffle Deck: resets deck counts to standard 2d6 distribution (36 cards)
⚠️ Does not clear recent-roll memory or 7 fairness/streak state

Undo Last: removes the most recent roll (rebuilds state from history)

What you’re seeing
Base distribution (deck-only)

Probability of each total based only on remaining deck counts:

𝑃
(
next
=
𝑡
)
=
cards remaining for 
𝑡
cards left in deck
P(next=t)=
cards left in deck
cards remaining for t
	​

Adjusted distribution (recent-roll + 7 fairness)

Starts from base, then applies:

a recent-roll penalty per total based on how often that total appears in the recent window

a 7 multiplier that depends on:

whether the current player has rolled “too many” or “too few” 7s

whether there is a streak of consecutive 7s by the same player

Then it renormalizes so probabilities sum to 1.

Important engine behavior

The deck has 36 total cards matching standard 2d6 totals:

2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:5, 9:4, 10:3, 11:2, 12:1

Pre-draw reshuffle rule:
If cards_left < 13, the engine reshuffles before applying the next roll.

If you manually input an “impossible” roll (no cards left for that total),
the engine reshuffles to recover.

Troubleshooting
“streamlit is not recognized” / command not found

Make sure your conda environment is activated:

conda activate catanbot
streamlit --version


If Streamlit isn’t installed in that environment:

pip install streamlit

The page loads but doesn’t update after clicking buttons

Usually a browser cache / Streamlit session hiccup:

refresh the page

make sure you’re clicking Apply Roll

check the terminal running Streamlit for errors

Next steps (planned)

Automatically ingest dice rolls from Colonist.io game chat (instead of manual entry)

Show an event log and per-player 7 stats over time

Compare “balanced dice” vs standard 2d6 distribution side-by-side