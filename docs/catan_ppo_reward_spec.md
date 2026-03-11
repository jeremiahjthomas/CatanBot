# Catan PPO Reward Shaping Specification

## Purpose
This document defines a practical reward-shaping scheme for a PPO-based 1v1 Catan bot trained under a Colonist.io-inspired ruleset. The goal is to improve sample efficiency while keeping the true objective aligned with winning the game.

The key design principle is:

```text
reward = terminal_reward + potential_based_shaping + small_event_rewards
```

This avoids the common failure mode where the agent learns to optimize hand-crafted shaping rewards instead of actual win probability.

---

## Ruleset Assumptions
This reward design assumes the following 1v1 Colonist.io-style rules:

- First to 15 victory points wins
- Balanced dice
- Discard on a 7 if a player has 10 or more resource cards
- Friendly Robber: a player can only be robbed if they have 3 or more visible victory points
- Development cards enabled
- Ports enabled
- No player-to-player trading
- Only bank and port trades

---

## High-Level Reward Formula
Use the following decomposition:

```text
r_t = r_terminal + lambda * (Phi(s_{t+1}) - Phi(s_t)) + r_small_events
```

Where:

- `r_terminal` is the dominant win/loss reward
- `Phi(s)` is a board-position potential function
- `r_small_events` contains modest event rewards and penalties
- `lambda` is a tuning constant, recommended initial value: `0.5`

---

## 1. Terminal Rewards
These should dominate all other rewards.

| Event | Reward |
|---|---:|
| Win game | +100 |
| Lose game | -100 |

Notes:
- Terminal rewards must be much larger than all shaping rewards.
- If illegal actions are possible, action masking is preferred over punishment.
- If illegal actions cannot be fully masked, use a small penalty such as `-2`.

---

## 2. Victory Point Rewards
Victory points are the cleanest dense signal because they align directly with the objective.

| Event | Reward |
|---|---:|
| Gain 1 visible VP | +6 |
| Gain hidden VP dev card | +5 |
| Opponent gains 1 visible VP | -6 |
| Opponent gains hidden VP dev card | -5 |
| Gain Longest Road | +12 |
| Lose Longest Road | -12 |
| Gain Largest Army | +12 |
| Lose Largest Army | -12 |

Implementation notes:
- Building a settlement usually implies `+6`
- Upgrading a settlement to a city gives net `+6`
- Hidden VP dev cards should be rewarded when acquired

---

## 3. Build Rewards
Do not make build rewards too large. The real value of a build should mostly come from the potential function and future win probability.

| Event | Reward |
|---|---:|
| Place road | +0.15 |
| Build settlement | +6 |
| Upgrade to city | +6 |
| Buy development card | +0.8 |

Notes:
- Roads should only get a tiny direct reward
- Over-rewarding roads will cause the bot to spam roads
- Settlements and cities already deserve strong rewards because they directly increase VP

---

## 4. Development Card Play Rewards
Development cards are strategically important in Colonist.io 1v1, but the reward should still be modest. Let the position improvement do most of the work.

| Event | Reward |
|---|---:|
| Play Knight | +0.8 |
| Play Road Building | +1.0 |
| Play Year of Plenty | +0.6 |
| Play Monopoly | +0.8 |

Additional Monopoly bonus:

```text
reward += 0.15 * net_cards_gained
cap total monopoly reward bonus at +1.2
```

Additional Knight bonus:
- If the knight advances toward Largest Army, add `+0.5`

Notes:
- VP dev cards are handled in the VP section when drawn
- Do not heavily reward dev-card play itself; reward the board improvement and win chance improvement instead

---

## 5. Robber Rewards and Penalties
Robber use in 1v1 should target what the opponent needs, not simply their highest pip tile or current visible hand. The shaping should reward strategic blocking.

### When placing the robber on the opponent

| Component | Reward |
|---|---:|
| Base valid robber move | +0.3 |
| Blocked production value | `+0.25 * blocked_pip_value` |
| Blocked bottleneck resource | +0.6 |
| Stolen card is immediately useful | +0.1 |
| Steal likely breaks opponent build | +0.1 |
| Clearly wasted robber move | up to -0.5 |

Recommended total cap for a robber move reward:

```text
cap total robber reward at around +2.0
```

### When the robber is placed on the agent

| Component | Penalty |
|---|---:|
| Blocked production value | `-0.25 * blocked_pip_value` |
| Blocked bottleneck resource | -0.6 |
| Stolen card | -0.15 |
| Stolen card breaks immediate build | -0.2 |

Recommended total cap for robber-on-self penalty:

```text
cap total robber penalty at around -2.0
```

Implementation notes:
- `blocked_pip_value` should reflect the pip strength of the blocked tile adjusted for settlement vs city and robber effects
- Bottleneck logic should identify the resource most important for the opponent’s likely next build

---

## 6. Discard Penalties
Discarding should be clearly negative but not catastrophic.

| Event | Penalty |
|---|---:|
| Discard on 7 | `-0.20 * cards_discarded` |
| Extra over-limit exposure | `-0.05 * (hand_before_discard - 9)` |

Example:
- If the agent discards 5 cards after holding 12 cards:

```text
penalty = -0.20 * 5 - 0.05 * (12 - 9)
        = -1.0 - 0.15
        = -1.15
```

Notes:
- Do not punish merely holding many cards too strongly
- Sometimes resource hoarding is correct for a city or multi-action swing turn

---

## 7. Trade Rewards
Trades should only receive reward when they improve action quality, not simply because a trade happened.

| Event | Reward |
|---|---:|
| Trade that immediately enables a strong build | +0.4 |
| Wasteful trade | -0.1 |

Examples of strong builds enabled by trade:
- settlement
- city
- development card
- immediate defensive move
- key tempo move

Most trade value should still come through the potential function.

---

## 8. Hand Management Helpers
These are optional tiny shaping terms.

| Event | Reward |
|---|---:|
| End turn build-ready | +0.1 |
| End turn above 9 cards without tactical justification | -0.1 |

Use these only as weak signals.

---

## 9. Resource Collection Rewards
In general, do not reward each resource card heavily. That can make the bot chase raw card count rather than winning.

Optional early-training helper:

| Event | Reward |
|---|---:|
| Gain one resource card | +0.02 |
| Opponent gains one resource card | -0.02 |

Recommendation:
- Use this only early if training is unstable
- Remove or reduce it later once the bot learns production value through the potential function

---

## 10. Potential Function Phi(s)
The potential function should encode positional strength in 1v1 Catan. This is where most strategic shaping should happen.

Recommended initial form:

```text
Phi(s) =
  0.20 * self_pip_production
- 0.20 * opp_pip_production
+ 0.80 * self_resource_diversity
- 0.80 * opp_resource_diversity
+ 1.50 * active_3to1_port
+ 2.00 * active_aligned_2to1_port
+ 0.25 * best_reachable_settlement_value
+ 0.60 * largest_army_edge
+ 0.40 * longest_road_edge
+ 1.20 * can_build_settlement_now
+ 1.20 * can_build_city_now
+ 0.80 * can_buy_dev_now
- 0.25 * self_blocked_pip_value
+ 0.25 * opp_blocked_pip_value
```

This should be normalized before use if feature scales differ too much.

### Feature definitions

#### self_pip_production / opp_pip_production
Use pip weights for number tokens:

| Number | Pip value |
|---|---:|
| 6, 8 | 5 |
| 5, 9 | 4 |
| 4, 10 | 3 |
| 3, 11 | 2 |
| 2, 12 | 1 |

Production contribution rules:
- settlement = 1x pip value
- city = 2x pip value
- blocked tile = 0

#### self_resource_diversity / opp_resource_diversity
A simple version:
- reward number of distinct resources a player can currently produce
- optionally add a bonus for access to all five resources

Example:

```text
resource_diversity = number_of_distinct_producible_resources
bonus_all_five = +2.0
```

#### active_3to1_port
Binary feature:
- `1` if player currently has access to a 3:1 port
- else `0`

#### active_aligned_2to1_port
Binary or scaled feature:
- `1` if player has a 2:1 port aligned with a resource they are likely to oversupply
- else `0`

#### best_reachable_settlement_value
Estimate the value of the best legal settlement site the player can still reasonably contest.
Suggested components:
- pip value
- diversity gain
- port access gain
- denial value to opponent
Discount by road distance.

#### largest_army_edge
Can be something like:

```text
largest_army_edge = self_knights_played - opp_knights_played
```

Possibly clipped or threshold-adjusted.

#### longest_road_edge
Can be something like:

```text
longest_road_edge = self_live_road_strength - opp_live_road_strength
```

This should reflect realistic road-race equity rather than raw road count only.

#### can_build_settlement_now / can_build_city_now / can_buy_dev_now
Binary indicators:
- `1` if the player can legally and immediately perform the action
- else `0`

#### self_blocked_pip_value / opp_blocked_pip_value
Total pip strength currently blocked by the robber for each player.

---

## 11. Recommended Full Reward Table

### Terminal
- Win: `+100`
- Loss: `-100`

### VP events
- Gain 1 visible VP: `+6`
- Gain hidden VP dev card: `+5`
- Opponent gains 1 visible VP: `-6`
- Opponent gains hidden VP dev card: `-5`
- Gain Longest Road: `+12`
- Lose Longest Road: `-12`
- Gain Largest Army: `+12`
- Lose Largest Army: `-12`

### Builds
- Place road: `+0.15`
- Build settlement: `+6`
- Upgrade to city: `+6`
- Buy dev card: `+0.8`

### Dev cards
- Play Knight: `+0.8`
- Knight advancing Largest Army race: `+0.5`
- Play Road Building: `+1.0`
- Play Year of Plenty: `+0.6`
- Play Monopoly: `+0.8 + 0.15 * net_cards_gained`, cap bonus at `+1.2`

### Robber
- Base valid robber move: `+0.3`
- Blocked production: `+0.25 * blocked_pip_value`
- Bottleneck block: `+0.6`
- Stolen useful card: `+0.1`
- Broke opponent build: `+0.1`
- Wasted robber move: up to `-0.5`
- Got robbed: `-0.15`
- Got robbed and immediate build broken: `-0.2`
- Own blocked production: `-0.25 * blocked_pip_value`
- Own blocked bottleneck resource: `-0.6`

### Discard
- Discard on 7: `-0.20 * cards_discarded`
- Over-limit exposure: `-0.05 * (hand_before_discard - 9)`

### Trade
- Trade enabling strong move: `+0.4`
- Wasteful trade: `-0.1`

### Tiny helpers
- End turn build-ready: `+0.1`
- End turn above 9 cards without justification: `-0.1`

### Potential shaping coefficient
- `lambda = 0.5`

---

## 12. What Not To Do
Avoid these reward-design mistakes:

1. Do not heavily reward roads.
   - The agent may learn to spam roads instead of winning.

2. Do not heavily reward raw resource gain.
   - The agent may chase cards rather than conversion into tempo and VP.

3. Do not over-reward playing development cards.
   - A dev card is only as good as the position change it creates.

4. Do not punish robber/discard events so much that the agent becomes too risk-averse.
   - Sometimes temporary exposure is strategically correct.

5. Do not reward “having lots of cards” as if it is always good.
   - Large hands can be useful, but they can also be dangerous under the discard rule.

---

## 13. Training Recommendations
Use a phased approach.

### Phase 1: Strong shaping
Use:
- terminal rewards
- VP rewards
- modest build rewards
- robber/discard shaping
- simple potential function

### Phase 2: Reduce shaping
Reduce hand-crafted event weights by about 30 to 50 percent.
This encourages PPO to rely more on actual value learning.

### Phase 3: Keep only the most reliable shaping
Keep:
- terminal rewards
- VP rewards
- robber/discard shaping
- potential-based shaping

Reduce or remove weak helpers such as per-resource rewards and tiny hand-management bonuses.

---

## 14. Suggested Implementation Notes for Claude
When converting this into code:

1. Implement rewards as a centralized table or config object.
2. Keep potential-based shaping separate from event rewards.
3. Log each reward component independently for debugging.
4. Store per-step reward breakdowns in rollout traces so training can be audited.
5. Make reward weights easy to tune without rewriting environment logic.
6. Prefer action masking for illegal actions instead of learning from penalties.
7. Make robber evaluation depend on opponent need, not just highest pip tile.
8. Track visible VP and hidden VP separately because Friendly Robber depends on visible VP only.

---

## 15. Suggested Next Files to Implement
A clean code implementation will likely want:

- `rewardConfig.ts` or `reward_config.py`
- `rewardCalculator.ts` or `reward_calculator.py`
- `potentialFunction.ts` or `potential_function.py`
- `robberEvaluation.ts` or `robber_evaluation.py`
- `vpTracker.ts` or `vp_tracker.py`

Optional:
- `rewardDebugLogger.ts`
- `rewardAblationConfig.ts`

---

## 16. Summary
This reward design aims to:
- keep winning as the dominant objective
- provide dense enough shaping for PPO to learn efficiently
- reflect actual Colonist.io 1v1 priorities
- avoid common reward hacking behaviors

The safest overall structure is:
- large terminal rewards
- medium VP-based rewards
- small event-based shaping
- strategic potential-based shaping

This should be treated as a strong starting point, not a final truth. Reward weights should be tuned after observing actual self-play behavior.
