# 1v1 Colonist.io Catan Initial Placements Guide

A practical research guide for building strong opening heuristics and PPO features for **1v1 Colonist.io Catan**.

---

## Table of Contents

1. [Why initial placement matters so much in 1v1](#why-initial-placement-matters-so-much-in-1v1)
2. [Important 1v1 Colonist.io rule context](#important-1v1-colonistio-rule-context)
3. [Core placement evaluation framework](#core-placement-evaluation-framework)
4. [The four main strategy archetypes](#the-four-main-strategy-archetypes)
5. [Best default strategy: Hybrid OWS](#best-default-strategy-hybrid-ows)
6. [Pure OWS](#pure-ows)
7. [Road Builder](#road-builder)
8. [Balanced](#balanced)
9. [Player 1 placement theory](#player-1-placement-theory)
10. [Player 2 placement theory](#player-2-placement-theory)
11. [How to react to opponent placement](#how-to-react-to-opponent-placement)
12. [Road direction and cutoff theory](#road-direction-and-cutoff-theory)
13. [Port valuation in 1v1](#port-valuation-in-1v1)
14. [Common placement mistakes](#common-placement-mistakes)
15. [PPO feature engineering and reward shaping](#ppo-feature-engineering-and-reward-shaping)
16. [Practical heuristics summary](#practical-heuristics-summary)
17. [Sources](#sources)

---

## Why initial placement matters so much in 1v1

In 1v1 Colonist.io, openings matter even more than in normal 4-player Catan.

That is because:

- there is only **one opponent**, so every denial matters more
- there is much more **open board space**, so road direction and race lines matter a lot
- there is much less trading, so your setup must be **self-sufficient**
- the ranked 1v1 format is played to **15 VP**, so your setup must support an actual long-game scoring plan, not just a decent early game

A weak opening in 1v1 often means one of two things:

- you get out-expanded and lose control of the board
- you build an engine that cannot actually close to 15 points

That is why initial placement is one of the most important things for a PPO model to learn well.

---

## Important 1v1 Colonist.io rule context

These format details shape what strong openings look like:

- Ranked 1v1 is played to **15 Victory Points**.
- Colonist 1v1 strategy guides emphasize that you generally cannot win without at least one **city**, and in practice you often need **Largest Army**, **Longest Road**, and/or **VP development cards** to close.
- Colonist 1v1 also uses **Friendly Robber** and **Balanced Dice**, which reduce some early-game punishment and variance.
- Initial placement still follows the normal Catan pattern: each player places **two settlements and two roads**, and you collect starting resources from the tiles around your **second settlement**.

This leads to one huge conclusion:

> In ranked 1v1, the best openings are not just “high production” openings. They are openings that convert production into a realistic 15-point path.

---

## Core placement evaluation framework

A strong opening should be evaluated across **six dimensions**, not just raw pips.

### 1. Production quality

Use pips as a first approximation of production strength.

Standard pip values:

- 6, 8 = 5 pips
- 5, 9 = 4 pips
- 4, 10 = 3 pips
- 3, 11 = 2 pips
- 2, 12 = 1 pip

Approximate settlement expected value:

```text
EV(settlement income per turn) ≈ (pip1 + pip2 + pip3) / 36
```

Example:

- A 6-5-9 corner has 5 + 4 + 4 = 13 pips
- EV ≈ 13/36 ≈ 0.36 cards per turn

This is not the whole story, but it is a good baseline.

### 2. Resource quality

Not all pips are equal.

In 1v1, **ore and wheat are disproportionately valuable** because:

- cities require **3 ore + 2 wheat**
- development cards require **1 ore + 1 wheat + 1 sheep**
- development cards are stronger in 1v1 because knights, VP cards, Road Building, and Monopoly all scale very well in a duel

Wood and brick are still important because they determine:

- how quickly you can expand
- whether you can win settlement races
- whether you can threaten or defend Longest Road
- whether your good spots are actually reachable

### 3. Number diversity

A setup with great pips but poor number spread can stall hard.

Example of bad diversity:

- two main resources both only on 6/8
- no production on 5/9/10/4 range

Balanced dice lowers variance somewhat, but strong number overlap can still make your setup clunky.

Good diversity means:

- your core resources are spread across multiple rolls
- you are not over-dependent on one token
- your opponent cannot cripple your engine by robbing one number or one hex

### 4. Resource coverage

In 1v1, missing a resource matters more because you cannot rely on table trading.

The usual order of preference is:

1. all five resources if possible
2. four resources with a clear port or bank-trade plan
3. three-resource specialization only if the engine is truly exceptional

### 5. Road utility

A road is not valuable just because it points somewhere.

A road is valuable if it does at least one of these:

- secures a premium future settlement spot
- threatens a cutoff
- protects your port access
- creates pressure on the opponent’s expansion route
- keeps your strategy mobile if your engine gets slowed

### 6. Close-out path

Ask this before finalizing placements:

> If this setup gets average rolls, how does it actually reach 15?

Possible close-out paths:

- city + dev card engine into Largest Army and VP cards
- settlement snowball into Longest Road, then pivot to city/dev
- flexible balanced setup that can take whichever line the game gives

If the answer is unclear, the opening is probably weaker than it looks.

---

## The four main strategy archetypes

There are four main opening archetypes worth explicitly modeling.

1. **Hybrid OWS**
2. **Pure OWS**
3. **Road Builder**
4. **Balanced**

For a PPO model, these are useful as both:

- human-interpretable strategic labels
- latent policy clusters that may correspond to different opening value functions

---

## Best default strategy: Hybrid OWS

### Definition

**Hybrid OWS** is the strongest all-around default strategy in ranked 1v1.

It means:

- prioritizing **ore + wheat + sheep** enough to support cities and development cards
- while still preserving **some mobility**, usually through:
  - a touch of wood or brick production, or
  - a realistic port line, or
  - a road race you can actually win

In simple terms:

> Hybrid OWS is an OWS engine with legs.

### Why it is the highest-probability strategy

Hybrid OWS tends to be the most consistent opening style across random boards because it matches what 1v1 asks for the most often.

It does three things at once:

#### 1. It funds cities

Cities are mandatory in practice because 15 VP is a long game and city efficiency is too important to ignore.

#### 2. It funds development cards

Development cards are especially strong in 1v1 because:

- knights control the robber and enable Largest Army
- VP cards help reach 15 without overbuilding the board
- Monopoly is stronger with only one opponent to track
- Road Building can swing races for ports and settlement spots
- Year of Plenty smooths missing-resource openings

#### 3. It avoids immobility

Pure OWS can fail because it has no ability to contest space. Hybrid OWS fixes that by keeping at least one real expansion route alive.

### Typical resource targets

As a rough directional guide, Hybrid OWS usually wants:

- strong **ore**
- strong **wheat**
- at least decent **sheep**
- at least light access to **wood or brick**, or a credible substitute via port

A very strong Hybrid OWS opening often looks like:

- one settlement that is city-quality, such as ore-wheat-sheep or ore-wheat-high pip
- one settlement that fixes mobility, such as wood-brick-wheat, wood-port-access, brick-port-access, or a strong 3:1 route

### Typical early plan

- get first city quickly
- buy dev cards consistently
- use limited roads efficiently, not greedily
- claim one or two high-value expansions or ports
- pressure Largest Army while keeping Longest Road as a possible side path

### Player 1 version

As Player 1, Hybrid OWS usually means:

- first pick a premium ore-wheat core if one exists
- then use the snake pick to patch mobility or coverage
- choose roads that either:
  - secure a needed third resource cluster, or
  - keep the opponent from boxing you in

### Player 2 version

As Player 2, Hybrid OWS often means:

- punish over-specialized P1 openings
- take the best remaining city core if P1 leaves it
- or deliberately take the board-control settlement first, then use the final pick to complete the engine

### Common mistakes

- taking “fake mobility” that does not actually reach a spot
- taking OWS numbers that are high pip but too concentrated on one token
- assuming a port solves everything when the road line is insecure
- ignoring brick completely on a board where races are sharp

### PPO interpretation

Hybrid OWS should likely be your model’s baseline preferred archetype when:

- ore+wheat EV is high
- sheep is present or reachable
- mobility is nonzero
- close-out path to 15 is clear

---

## Pure OWS

### Definition

**Pure OWS** means prioritizing ore, wheat, and sheep aggressively, often with minimal or zero wood/brick production.

This is the classic:

- city fast
- dev spam
- Largest Army + VP cards
- use tactical devs or bank trades to place only the minimum roads and settlements needed

### When it is good

Pure OWS is best when:

- the ore and wheat spots are outstanding
- sheep support is good enough for repeated dev buying
- the board gives enough natural safety that lack of roads will not immediately punish you
- the opponent’s likely counterplay is slow or awkward

### When it is bad

Pure OWS is dangerous when:

- your wheat or ore comes from only one main hex
- the opponent can easily block your only expansion line
- you lack a fast port route
- you cannot realistically place future settlements without miracle dev cards

### Typical resource profile

Ideal Pure OWS generally means:

- premium ore
- premium wheat
- usable sheep
- little or no wood/brick

### Typical early plan

- rush first city
- buy dev cards early and often
- use knights to suppress opponent tempo
- get Largest Army
- win with a mix of cities, Army, and hidden VP cards

### Main advantage

The strength of Pure OWS is efficiency.

Every good roll converts directly into the highest-value actions in the game.

### Main weakness

The weakness is geometry.

If you cannot move, the board can beat your engine.

### PPO interpretation

A PPO model should only strongly favor Pure OWS when:

- the OWS engine is clearly dominant in EV
- positional risk is low
- road dependence is unusually low
- opponent denial options are limited

Otherwise, Hybrid OWS should usually dominate Pure OWS.

---

## Road Builder

### Definition

**Road Builder** is a road-and-settlement-first strategy focused on:

- wood and brick production
- fast expansion
- early settlement count
- cutoff pressure
- Longest Road pressure

It is the most board-control-oriented of the four archetypes.

### What it tries to do

Road Builder aims to:

- claim more space than the opponent
- deny their best future spots
- force them into awkward bank trades or weak ports
- convert early map control into Longest Road and settlement points

### When it is good

Road Builder improves when:

- wood and brick are premium resources on this board
- there are sharp chokepoints
- one player can realistically seal off multiple future intersections
- the opponent is drifting into a greedy engine opening

### Why it is not the best default in 1v1

The problem is that 15 VP is a lot.

Even if you get many settlements and Longest Road, you often still need:

- at least one city
- and often more than one extra non-board point source

So Road Builder often must pivot later into:

- wheat/ore city building
- dev cards
- or port-based conversion

### Typical resource targets

Road Builder generally wants:

- strong wood
- strong brick
- enough wheat to keep settlement tempo going
- enough sheep to actually place settlements
- some path to ore later for at least one city

### Typical early plan

- expand first
- threaten cutoffs immediately
- take contested spots before starting dev/card engine
- use map pressure to force inefficient opponent lines

### Common mistakes

- overcommitting to roads without a finish plan
- winning the board but losing the game because the engine is too weak
- ignoring ore too much
- building roads toward spots the opponent can still beat you to

### PPO interpretation

Road Builder should be valued highly when the board contains:

- strong cutoff lines
- scarce brick access
- asymmetric road races
- port routes you can lock before the opponent

But its value should be penalized if the resulting setup has no credible 15-point conversion path.

---

## Balanced

### Definition

**Balanced** means trying to cover all five resources, maintain number diversity, and preserve flexibility.

This is the least fragile opening style.

### What it tries to do

Balanced aims to:

- reduce missing-resource stalls
- keep many lines open
- adapt after seeing early rolls, robber pressure, and opponent commitment

### When it is good

Balanced is often correct when:

- the board has no elite OWS core
- the road board is unclear
- the opponent’s picks force you into patching coverage instead of specializing
- ports are mediocre and bank conversion matters more

### Main advantage

The advantage is consistency.

Balanced openings rarely feel amazing, but they also avoid many dead states.

### Main weakness

The weakness is lack of efficiency.

If your balanced setup gives up too much ore+wheat quality, you may simply lose to a stronger city/dev engine.

### Typical early plan

- expand naturally
- evaluate whether the game is becoming city/dev heavy or road heavy
- pivot based on what the board gives

### Common mistakes

- confusing “balanced” with “low-quality but broad”
- taking all five resources but with terrible pips
- building a flexible setup that is not actually strong at anything

### PPO interpretation

Balanced should be favored as the fallback policy when:

- specialization opportunities are weak
- resource coverage is otherwise poor
- board control contests are uncertain
- opponent commitment leaves a broad, consistent setup as the best response

---

## Player 1 placement theory

Player 1 has first access to the board’s best single spot.

That gives P1 one huge advantage:

- the ability to define the game’s opening center of gravity

But it also creates one big challenge:

- after the first settlement, P2 gets two consecutive placements in the snake order

So P1 must think not just:

> What is the best first spot?

but also:

> What will still be available after P2 gets two picks?

### P1 priority order

A strong P1 thought process is:

1. identify the board’s best overall settlement
2. identify which complementary resources are most likely to disappear in the snake turn
3. decide whether your first road should preserve a fallback line if P2 double-blocks your ideal plan

### P1 strategic styles by archetype

#### P1 Hybrid OWS

Take the best city core first if available.

Then ask:

- can I still get mobility later?
- if not, should I lower my first-pick greed slightly and take the more complete first settlement?

#### P1 Pure OWS

Only do this if the premium OWS spot is truly exceptional and the board will still leave you a viable second settlement.

#### P1 Road Builder

This works best if the board has a clearly dominant expansion skeleton and you can take the anchor point first.

#### P1 Balanced

Use this when the board does not support a clearly winning specialization and the safe, broad setup is best.

### P1 main trap

The biggest P1 opening mistake is:

> taking the best first settlement in isolation, instead of the best two-settlement package after snake exposure

Sometimes the best single settlement leads to a bad overall opening because P2 can take both of your needed follow-ups.

---

## Player 2 placement theory

Player 2 has one of the strongest tactical tools in the game:

- the **double placement** in the middle of the snake order

That means P2 is often less about “taking the best spot” and more about **redefining the board**.

### P2 advantages

P2 can:

- deny P1’s natural complement
- pair two spots together immediately
- shape the board’s road races
- take both a strong settlement and a tactical denial settlement

### P2 priority order

A strong P2 thought process is:

1. understand what P1 is trying to become
2. identify the key missing piece in P1’s opening
3. decide whether it is stronger to:
   - deny that piece, or
   - ignore denial and assemble an even stronger own package

### P2 common tactical patterns

#### Pattern 1: deny the patch

If P1 takes an OWS core with weak mobility, P2 can take the wood/brick patch or the port line that would have stabilized it.

#### Pattern 2: deny the close-out path

If P1 is building Road Builder, P2 can take the only good ore/wheat follow-up so P1 struggles to transition later.

#### Pattern 3: take the pair

Sometimes the best P2 move is to ignore P1 and use the double pick to assemble a complete two-settlement package with better synergy.

### P2 main trap

The biggest P2 mistake is over-denial.

If you spend both picks hurting P1 but do not build a strong own engine, you may still lose.

The right question is not:

> How do I ruin P1?

It is:

> How do I improve my own opening the most, while also making P1 uncomfortable?

---

## How to react to opponent placement

This section matters a lot for both human play and PPO policy learning.

### If opponent opens greedy OWS

Common signs:

- very high ore/wheat concentration
- poor wood/brick access
- road lines that rely on one corridor

Best responses:

- take their mobility patch
- take the port that would fix their weakness
- force faster board races
- rob their single best wheat or ore source later

### If opponent opens Road Builder

Common signs:

- strong wood/brick concentration
- roads pointed at clear future intersections
- lower ore quality

Best responses:

- take the transition spot they need for wheat/ore
- break or race their corridor early
- avoid leaving them easy Longest Road structures
- make sure your own setup still closes efficiently

### If opponent opens Balanced

Common signs:

- all five resources
- broad but not hyper-optimized pips
- several pivot options

Best responses:

- be more efficient if the board allows it
- punish weak specialization by taking the best engine package
- do not give them a free premium port

### If opponent opens fragile number overlap

Common signs:

- too much tied to one number token
- most economy running through one main hex

Best responses:

- prefer denial lines over small EV gains
- later robber placement becomes much stronger
- do not accidentally let them diversify on the snake or next settlement

---

## Road direction and cutoff theory

A road should be judged by future board control, not by aesthetics.

### Good road directions

A good starting road usually does one of these:

- points to an uncontested premium spot
- creates a race the opponent cannot win cleanly
- threatens to cut across an area the opponent wants
- moves you toward a strong port that fixes your setup
- preserves optionality between two future intersections

### Bad road directions

A bad starting road often:

- points to a spot the opponent can take first anyway
- heads toward a weak intersection just to “do something”
- commits to a line that only works if multiple unlikely things happen
- leaves no backup if the opponent blocks you once

### Cutoff logic

A cutoff is powerful in 1v1 because there is only one opponent.

If you seal one lane, you are not just reducing competition. You may be eliminating their entire plan.

Key cutoff ideas:

- cutoffs are strongest when they deny a **missing resource patch**
- cutoffs are stronger when they attack a **port route**
- cutoffs are stronger when the opponent is already specialized
- cutoffs are less useful if they cost too much of your own setup quality

### PPO interpretation

Road direction should not be learned only from local geometry. It should be connected to:

- future reachable intersections
- opponent reachable intersections
- port access timing
- cutoff potential
- road race win probability

---

## Port valuation in 1v1

Ports are more important in 1v1 than many players think.

Because table trading barely exists, ports often replace inter-player trade as the main conversion outlet.

### 3:1 ports

These are often excellent in 1v1 because they smooth awkward but high-production setups.

A 3:1 port becomes especially strong when:

- your engine is high production but uneven
- you are OWS-heavy and can dump excess sheep or ore later
- your setup is broad enough to generate frequent mismatched hands

### 2:1 ports

These are extremely strong when tied to a dominant production resource.

Examples:

- ore port with heavy ore setup
- wheat port with city-heavy build
- wood or brick port with Road Builder setup

### Common port mistake

Players often overvalue a port that they cannot safely reach.

A port is only valuable if:

- the route is realistic
- the resource feeding it is meaningful
- getting there does not destroy your main plan

### Port rule of thumb

A port should be treated as a resource substitute only when the access line is genuinely credible.

---

## Common placement mistakes

Here are the biggest mistakes to avoid or penalize in training.

### 1. Valuing pips over plan

A setup can have lots of pips and still be weak if it has no clean path to 15.

### 2. Taking fake all-five coverage

All five resources with terrible quality is often worse than a four-resource setup with a strong engine and a real port.

### 3. Overcommitting to pure OWS

Pure OWS is powerful but easy to overforce.

### 4. Building roads with no purpose

Every starting road should have a strategic job.

### 5. Ignoring opponent patch points

In 1v1, stopping the opponent from fixing their weakness is often as important as improving your own pips.

### 6. Over-denying as P2

If your opening becomes weak while you try to hurt P1, you may still lose.

### 7. No transition plan in Road Builder

Road-heavy setups must still plan how to reach the final points.

### 8. Relying on one hex too much

Single-hex dependence makes robber pressure devastating.

### 9. Assuming a port solves everything

Unreached ports do nothing.

### 10. Ignoring number diversity

Two great numbers are not always better than three good numbers with spread.

---

## PPO feature engineering and reward shaping

This is the most useful section if you are training a Catan agent.

The goal is to help the PPO model learn that initial placement is not just a static pip maximization problem.

It is a **structured long-horizon decision**.

### A. Recommended opening-state features

Your model should see features that describe both local settlement quality and long-run strategic meaning.

#### 1. Raw settlement features

For each candidate intersection:

- total pip count
- pip count by resource
- number diversity score
- count of unique resources
- whether touching desert or poor tokens

#### 2. Resource composition features

For a two-settlement package:

- total pips by resource across both settlements
- indicator for all five resources
- indicator for four resources with port access
- OWS score = ore + wheat + sheep weighted sum
- road score = wood + brick weighted sum
- city score = weighted ore+wheat score
- dev score = weighted ore+wheat+sheep score

#### 3. Structural features

- number of reachable future intersections within 1, 2, 3 roads
- quality of those reachable intersections
- whether roads contest or secure those spots
- minimum road distance to nearest 3:1 port
- minimum road distance to each 2:1 port
- best cutoff potential from each starting road

#### 4. Opponent-relative features

- overlap in target intersections
- whether candidate denies opponent’s missing resource
- whether candidate steals opponent’s best port line
- whether candidate increases opponent dependency on one hex
- settlement race win/loss estimate for key future spots

#### 5. Strategic archetype features

- Hybrid OWS similarity score
- Pure OWS similarity score
- Road Builder similarity score
- Balanced similarity score

These can be hand-engineered labels or learned embeddings.

### B. Good reward shaping ideas

For opening placement, reward shaping should encourage **good long-run structure**, not just raw pips.

#### Positive shaping terms

You can reward:

- higher total pip EV
- high ore+wheat score when supported by a finish plan
- all-five coverage when quality is acceptable
- strong number diversity
- reachable premium expansion spots
- secure port access
- denying opponent’s critical patch resource
- creating a favorable road race
- opening lines that support city timing and dev timing

#### Negative shaping terms

You can penalize:

- high dependence on one hex
- no realistic expansion route
- missing critical resources with no port plan
- fake port access
- roads leading to dead ends
- opening packages with no credible route to 15 VP
- over-specialization when the board geometry punishes it

### C. Important caution on rewards

Do **not** reward simple OWS too hard in every position.

If you over-reward ore+wheat+sheep pips blindly, the policy may collapse into greedy but fragile placements.

Instead, use conditional rewards such as:

```text
OWS bonus only if mobility_score >= threshold
Port bonus only if access_probability >= threshold
Road bonus only if future_spot_quality >= threshold
```

That helps the model prefer **Hybrid OWS** over bad Pure OWS.

### D. Multi-step opening reward design

You can shape the opening in stages.

#### After first settlement

Reward:

- strong core value
- flexibility under snake exposure
- preservation of good second-settlement complements

Penalize:

- openings that leave only one viable complement and let opponent steal it easily

#### After second settlement

Reward:

- full package synergy
- strategic coherence
- clear path to city/dev or road expansion

Penalize:

- disconnected package
- unresolved missing resource problem
- weak close-out potential

#### After initial roads

Reward:

- secure reachability
- cutoff leverage
- strong port line
- multiple future options

Penalize:

- dead-end roads
- roads into already-lost races
- roads that do not improve reachable EV

### E. Archetype-aware training idea

A strong approach is to explicitly classify each opening package into one of the four archetypes and then use:

- archetype-conditioned value estimates
- archetype-aware auxiliary losses
- imitation labels from strong human heuristics for openings only

This can help PPO learn that the same pip count means different things depending on structure.

### F. Suggested scalar heuristics for bootstrapping

Here is a simple handcrafted starting score for candidate two-settlement packages:

```text
opening_score =
    1.0 * total_pips
  + 1.3 * city_score
  + 1.1 * dev_score
  + 0.7 * number_diversity
  + 0.8 * resource_coverage
  + 0.9 * mobility_score
  + 0.8 * cutoff_score
  + 0.6 * port_access_score
  - 1.0 * single_hex_dependence
  - 0.8 * dead_road_penalty
  - 1.2 * no_closeout_penalty
```

You would tune these weights empirically, but this is a useful first prior.

### G. Best modeling lesson

The most important lesson for a PPO model is:

> Initial placement value is joint, not local.

The model should learn to value:

- the two-settlement package
- the road directions
- the opponent reaction space
- and the eventual 15-point conversion plan

not just the first spot’s pips.

---

## Practical heuristics summary

If you want a compact ruleset for your repo, use this:

### Best default

- Prefer **Hybrid OWS** when strong ore+wheat exists and mobility is still real.

### Pure OWS only when

- the OWS engine is clearly dominant
- the board geometry is forgiving
- the opponent cannot easily steal your patch or lock your route

### Prefer Road Builder when

- wood/brick are elite
- there are strong cutoffs
- you can still pivot later into city/dev scoring

### Prefer Balanced when

- the board does not offer a clean specialization
- all-five coverage plus decent pips beats awkward greed

### As Player 1

- optimize the **two-settlement package**, not just the first pick

### As Player 2

- use the double pick to either:
  - deny a key patch, or
  - build the strongest own pair

### Always check

- Can this setup get a city?
- Can this setup buy devs efficiently?
- Can this setup move on the board?
- Can this setup reach 15, not just 8 or 10?

---

## Sources

Primary sources used in the research summary:

- Colonist.io, **Ranked 1v1 - A Comprehensive Strategy Guide**  
  https://blog.colonist.io/ranked-1v1-comprehensive-strategy-guide-colonist-io/

- Colonist.io, **Ranked 1v1 Games Strategy**  
  https://blog.colonist.io/ranked-1v1-strategy/

- Colonist.io, **Colonist Rules - Base Game**  
  https://colonist.io/catan-rules

- Colonist.io, **The Best Catan Starting Strategies**  
  https://blog.colonist.io/guide-to-catan-starting-strategies/

- Colonist.io, **How to Gain Victory Points and Win in Catan**  
  https://blog.colonist.io/how-to-gain-victory-points-in-catan/

- Rempton Games, **The Ultimate Catan Strategy Guide – Top Tips to Win More at Catan**  
  https://remptongames.com/2021/08/08/the-ultimate-catan-strategy-guide-top-tips-to-win-more-at-catan/

- Reddit discussion, **1v1 strategy tips?**  
  https://www.reddit.com/r/Colonist/comments/1fhe8df/1v1_strategy_tips/

Relevant video/search leads referenced in the research process included high-level 1v1 Colonist strategy and OWS vs Hybrid OWS commentary, including titles such as:

- **I Solved Catan's Most Popular Gamemode**
- **Catan Placements | Hybrid OWS (Strongest Strategy)**
- **THE PERFECT 1v1 START!!!**
- **Strategy of Catan - OWS vs Hybrid OWS**

---

## One-sentence takeaway

In ranked 1v1 Colonist.io, the strongest initial placements usually are not the greediest ones, but the ones that combine a strong **ore-wheat core** with enough **mobility, denial, and future structure** to actually convert into a reliable 15-point win path.
