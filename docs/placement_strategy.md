# Basic 1v1 Catan Strategy for PPO Training

This document encodes a simple, rule-based strategy inspired by high‑level 1v1 Catan play, focusing on **initial placements** and **pip thresholds** by resource to guide your PPO agent’s behavior.[page:3][page:2][page:1]

---

## 1. Core Strategic Principles

- Prioritize ore–wheat–sheep (OWS) over everything else in 1v1.[page:3][page:2]
- Strong wheat and ore, with enough sheep for development cards, is more valuable than “nice looking” wood/brick starts.[page:3]
- Dev‑first, road‑second: in strong OWS setups, prefer devs and city timing rather than racing purely for road.[page:3][page:2]
- Brick and wood are often acquired later, or via steals when you control knights and the robber.[page:3]

We measure number quality using **pips**:

- 6 or 8 → 5 pips  
- 5 or 9 → 4 pips  
- 4 or 10 → 3 pips  
- 3 or 11 → 2 pips  
- 2 or 12 → 1 pip

---

## 2. Scoring a Settlement (Single Vertex)

For a candidate settlement vertex, define:

- \(P_w\) = total wheat pips on the three adjacent hexes  
- \(P_o\) = total ore pips  
- \(P_s\) = total sheep pips  
- \(P_b\) = total brick pips  
- \(P_{wd}\) = total wood pips

### 2.1 Base score

Use this weighted sum:

\[
Score = 3.0 P_w + 2.5 P_o + 2.0 P_s + 1.3 P_b + 1.3 P_{wd}
\]

This creates an explicit bias towards wheat and ore while still valuing brick and wood.[page:3]

### 2.2 Bonuses

Add discrete bonuses:

- +2 if the vertex has **both wheat and ore** (OWS core).[page:3]  
- +1 if the vertex touches **two wheat tiles**.  
- +1 if the vertex touches **two ore tiles**.  
- +2 if it has a **6 or 8 wheat** plus **any ore**.[page:3]  
- +1 if a **wheat or sheep port** is reachable in ≤ 2 roads.[page:3]

This scoring function should make your bot prefer classic OWS setups like “8–4–10 with ore and sheep” over flat “balanced” starts with slightly better brick/wood.[page:3]

---

## 3. Choosing Initial Placements (Two Settlements)

In 1v1, you want to evaluate **pairs** of starting settlements under the turn order rules.

### 3.1 Combined pip totals

For a pair \( (S_1, S_2) \), define:

- \(P_w^{tot} = P_w(S_1) + P_w(S_2)\)  
- \(P_o^{tot} = P_o(S_1) + P_o(S_2)\)  
- \(P_s^{tot} = P_s(S_1) + P_s(S_2)\)  
- \(P_b^{tot} = P_b(S_1) + P_b(S_2)\)  
- \(P_{wd}^{tot} = P_{wd}(S_1) + P_{wd}(S_2)\)

### 3.2 Combo scoring function

Define a **combo score** for the pair:

\[
ComboScore = 3 P_w^{tot} + 2.5 P_o^{tot} + 2 P_s^{tot} + 0.8 P_b^{tot} + 0.8 P_{wd}^{tot} + 3 \cdot PortBonus
\]

where:

- PortBonus = 1 if either settlement can reach a **wheat or sheep port** within 2 roads; 0 otherwise.[page:3]

This again emphasizes total OWS strength over brick/wood, with a strong bump for relevant ports.

### 3.3 Hard OWS preference rule

If there exists a pair \( (S_1, S_2) \) such that:

- \(P_w^{tot} ≥ 9\)  
- \(P_o^{tot} ≥ 8\)  
- \(P_s^{tot} ≥ 6\)

then **force** selection of that pair, even if brick/wood are weaker.[page:3]  

This encodes the “strongest setup” heuristic: locking a dominant OWS core is more important than having perfect early roads.[page:3]

### 3.4 First settlement

When choosing the **first** settlement:

- Compute ComboScore for all legal two‑settlement combinations compatible with turn order.  
- Prefer combinations where the first settlement is on a **OWS vertex** (wheat + ore + sheep) even if an alternative wheat spot offers slightly better wood/brick.[page:3]  
- If there is a unique high‑pip wheat tile (e.g., 8‑wheat), prefer to anchor the first settlement on that tile with ore/sheep rather than chasing a “balanced” 5‑9‑10 style spot.[page:3]

### 3.5 Second settlement and road direction

After placing S1:

- Evaluate all legal S2 placements under turn order, and recompute ComboScore(S1, S2).  
- For each S2, also compute:
  - Distance to wheat/sheep/3:1 port.  
  - Whether S2 allows a **“plow”** (cut the opponent off from an obvious key target like their only 6 wheat or only ore).[page:2]

Define:

- PlowChance = 1 if a 2‑road sequence from S1 or S2 can deny a critical opponent target; otherwise 0.[page:2]  
- BlockedByOpponent = penalty estimate if S2 is likely to be stolen by opponent before you can connect.

Then pick S2 that maximizes:

\[
Score_{S2} = ComboScore(S1, S2) + 2 \cdot PlowChance - BlockedByOpponent
\]

Initial road direction should:

- Point towards the **key port** (wheat, sheep, or strong 3:1) **or** towards a high‑value contested hex.[page:2][page:3]  
- Avoid pointing at low‑value spots just to grab an extra settlement if it doesn’t support your OWS core.

---

## 4. Plan Classification via Pip Thresholds

After initial placements, classify the bot’s **macro plan** based on total pips.

### 4.1 OWS dev plan (default preference)

Enter **OWS dev mode** if:

- Total wheat pips \(P_w^{tot} ≥ 7\)  
- Total ore pips \(P_o^{tot} ≥ 6\)  
- Total sheep pips \(P_s^{tot} ≥ 5\)  
- At most one of brick or wood has ≥ 5 pips (road game is weaker).[page:3][page:2]

In OWS dev mode:

- Prioritize:
  - Buying **development cards** when hand contains wheat + ore + sheep.  
  - **City** the best OWS spot as soon as safely possible.[page:3]
- Long‑run target pip mix (including first city) roughly:
  - Wheat: 10–12 pips  
  - Ore: 8–10 pips  
  - Sheep: 7–9 pips  
  - Brick/wood: 4–6 pips each (can be improved later or via steals).[page:3]

### 4.2 Road/port plan

Enter **road/port mode** if:

- Brick pips \(P_b^{tot} ≥ 7\)  
- Wood pips \(P_{wd}^{tot} ≥ 7\)  
- Wheat pips \(P_w^{tot} ≤ 5\)  
- Ore pips \(P_o^{tot} ≤ 4\).[page:2]

In road/port mode:

- Race for:
  - 3:1 or relevant resource port.  
  - High‑production new spots that fix wheat weakness.  
- Limit early dev buys unless combined wheat+ore pips ≥ 6.[page:2]

Long‑run target thresholds:

- Brick: 9–11 pips  
- Wood: 9–11 pips  
- Wheat: ≥ 6 pips by the first or second expansion  
- Ore: any access is a bonus, not mandatory early.

---

## 5. City vs Development Card Timing

Use sevens and hand exposure to decide between **city** and **dev**.[page:3]

Define:

- \(r_7\) = number of 7s rolled  
- \(r_{tot}\) = total rolls  
- \(f_7 = r_7 / r_{tot}\) = observed 7 frequency

Heuristic:

- If \(f_7 < 0.12\) (7s are under‑rolled) and the bot has ~4 cards including a valid city recipe (e.g., wheat + ore + ore + extra card):
  - Prefer **city first** on the core OWS spot.[page:3]
- If \(f_7 ≥ 0.12\) or hand size ≥ 7 cards:
  - Prefer **dev first**, especially when city would leave 8–9 cards and no knight in hand.[page:3]

Additional safety rules:

- If the bot has 2 knights already, a city is safer; he often waits for at least 2 knights before rushing into a second city.[page:3]  
- If the opponent has only one wheat spot, prioritize devs to keep that wheat blocked and delay their city; city afterwards.[page:3]

---

## 6. Opponent‑Aware Heuristics

Track opponent’s pip totals after placements:

- OppWheatPips  
- OppOrePips

### 6.1 Choke points for the robber

- If the opponent has exactly one wheat vertex, mark its main wheat hex as **choke wheat**.  
  - Default to blocking choke wheat unless:
    - They also have only one similar‑pip ore tile, and they are in clear city range; in that case, sometimes blocking ore is higher value.[page:3][page:2]

### 6.2 Steal priorities

In OWS dev mode:

- When stealing, prefer to hit **brick or wood** from the opponent if you already have strong ore.[page:3]  
- When the opponent is obviously in road/port mode (high brick/wood, low ore), your bot should:
  - Maintain **knight** lead for largest army.  
  - Block their wheat/ore so they cannot pivot into devs.[page:2][page:3]

---

## 7. Interface to PPO / Claude

This section explains how to use these rules as a prior for training.

### 7.1 Features for the neural net

For each state (especially setup):

- For each candidate settlement:
  - \(P_w, P_o, P_s, P_b, P_{wd}\)  
  - Vertex bonuses: hasWheat, hasOre, hasSheep, has6or8Wheat, hasPortWithin2  
  - Single‑vertex **Score** as defined above.
- For each pair of settlements:
  - Combined pip totals and **ComboScore**.  
  - Flags for OWS thresholds satisfied.
- Opponent features:
  - OppWheatPips, OppOrePips.  
  - OppHasSingleWheat, OppHasSingleOre.

### 7.2 Reward shaping suggestions

For PPO:

- Add **positive reward** when:
  - Initial pair meets OWS thresholds (OWS dev mode).  
  - First city is placed on the highest OWS vertex before opponent’s first city.  
  - The bot secures knight lead while in OWS dev mode.[page:3][page:2]
- Add **negative reward** when:
  - Initial placements leave total wheat pips < 6 and total ore pips < 6.  
  - The chosen start has lower ComboScore than an available OWS‑strong pair by a margin (i.e., it “ignored” a much better OWS setup).

### 7.3 Natural language prompt for Claude

You can summarize to Claude roughly like this (paraphrased for your system):

- “Prefer setups with high total wheat, ore, and sheep pips (OWS), even if brick/wood are weaker.”  
- “Score each settlement and pair using pip‑weighted formulas and pick the highest‑scoring OWS‑heavy pair.”  
- “Classify your plan as OWS dev or road/port using simple pip thresholds, and then prioritize devs/cities or roads/ports accordingly.”  
- “Use robber and dev cards to attack the opponent’s only wheat or ore, and bias steals towards brick/wood when you are OWS‑heavy.”

---

If you share your exact environment details (board generation, ports, trade rules), I can tighten these thresholds and scoring weights for that specific 1v1 variant.
