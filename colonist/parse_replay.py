"""
parse_replay.py

Parses a captured Colonist.io replay JSON into a structured game timeline.
Outputs a turn-by-turn log and an actions.json with (state, action) pairs.

Usage:
    python parse_replay.py replay_207429263_gamedata.json
    python parse_replay.py replay_207429263_gamedata.json --out actions_207429263.json
"""

import json
import sys
import argparse
from copy import deepcopy
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOURCE_NAMES = {1: "wood", 2: "brick", 3: "sheep", 4: "wheat", 5: "ore"}
BUILDING_NAMES = {1: "settlement", 2: "city"}
PIECE_NAMES    = {0: "road", 2: "settlement", 3: "city"}
DEV_CARD_NAMES = {
    0: "knight", 1: "road_building", 2: "year_of_plenty",
    3: "monopoly", 4: "vp", 5: "vp", 6: "vp", 7: "vp",
    10: "knight", 11: "knight",
}
ACHIEVEMENT_NAMES = {0: "longest_road", 1: "largest_army"}

LOG_TYPE_NAMES = {
    0: "connected", 1: "game_start", 4: "build", 5: "build_road",
    10: "dice_roll", 11: "robber_placed", 14: "steal",
    15: "receive_cards", 20: "play_dev_card", 24: "game_init",
    44: "end_turn", 45: "game_over", 47: "receive_resources",
    49: "robber_blocks", 55: "receive_cards_multi", 60: "discard",
    66: "gain_achievement", 68: "lose_achievement", 74: "pass",
    86: "monopoly_steal", 116: "trade", 130: "bank_init",
}

# actionState codes from currentState
ACTION_STATE_NAMES = {
    0: "main_phase",
    1: "place_road",
    3: "place_settlement",
    24: "roll_or_play",
    28: "discard",
    30: "move_robber",
    31: "steal",
}

# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def empty_state(play_order):
    return {
        "turn": 0,
        "current_player": play_order[0],
        "play_order": play_order,
        "action_state": None,
        "corners": {},        # vertexId -> {owner, buildingType}
        "edges": {},          # edgeId   -> {owner}
        "player_resources": {},  # playerColor -> [list of resource enums]
        "player_vp": {},         # playerColor -> int
        "bank": {i: 19 for i in range(1, 6)},  # resource -> count
        "robber_tile": 0,
        "dev_cards": {},      # playerColor -> {played: n, total: n}
        "largest_army": None,
        "longest_road": None,
        "completed_turns": 0,
    }


def deep_merge(base: dict, diff: dict) -> dict:
    """Recursively merge diff into base (modifies base in place)."""
    for k, v in diff.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def apply_event(state: dict, event: dict) -> list:
    """
    Apply one event's stateChange to the cumulative state.
    Returns a list of action records extracted from this event.
    """
    sc = event.get("stateChange", {})
    actions = []

    # --- Map state ---
    for vid, cdata in sc.get("mapState", {}).get("tileCornerStates", {}).items():
        if vid not in state["corners"]:
            state["corners"][vid] = {}
        deep_merge(state["corners"][vid], cdata)

    for eid, edata in sc.get("mapState", {}).get("tileEdgeStates", {}).items():
        if eid not in state["edges"]:
            state["edges"][eid] = {}
        deep_merge(state["edges"][eid], edata)

    # --- Player states ---
    for color_str, pdata in sc.get("playerStates", {}).items():
        color = int(color_str)
        if "resourceCards" in pdata:
            state["player_resources"][color] = pdata["resourceCards"].get("cards", [])
        if "victoryPointsState" in pdata:
            # Merge VP diff into the running breakdown, then recompute total
            if color not in state.get("player_vp_breakdown", {}):
                state.setdefault("player_vp_breakdown", {})[color] = {}
            state["player_vp_breakdown"][color].update(pdata["victoryPointsState"])
            state["player_vp"][color] = sum(state["player_vp_breakdown"][color].values())

    # --- Bank state ---
    for res_str, count in sc.get("bankState", {}).get("resourceCards", {}).items():
        state["bank"][int(res_str)] = count

    # --- Dice state ---
    dice = sc.get("diceState", {})
    if dice.get("diceThrown"):
        d1 = dice.get("dice1", 0)
        d2 = dice.get("dice2", 0)
        state["last_dice"] = (d1, d2)

    # --- Current state ---
    cs = sc.get("currentState", {})
    if "currentTurnPlayerColor" in cs:
        state["current_player"] = cs["currentTurnPlayerColor"]
    if "actionState" in cs:
        state["action_state"] = cs["actionState"]
    if "completedTurns" in cs:
        state["completed_turns"] = cs["completedTurns"]
        state["turn"] = cs["completedTurns"]

    # --- Robber ---
    rs = sc.get("mechanicRobberState", {})
    if "locationTileIndex" in rs:
        state["robber_tile"] = rs["locationTileIndex"]

    # --- Achievements ---
    for color_str in sc.get("mechanicLargestArmyState", {}):
        state["largest_army"] = int(color_str)
    for color_str in sc.get("mechanicLongestRoadState", {}):
        state["longest_road"] = int(color_str)

    # --- Parse actions from game log ---
    for log_idx, log_entry in sc.get("gameLogState", {}).items():
        text = log_entry.get("text", {})
        if not isinstance(text, dict):
            continue
        ltype = text.get("type")
        player = text.get("playerColor") or log_entry.get("from")
        action = _parse_log_action(ltype, text, player, state)
        if action:
            actions.append(action)

    return actions


def _parse_log_action(ltype, text, player, state):
    """Convert a gameLog text entry into a structured action dict."""
    if ltype == 10:   # dice roll
        return {
            "type": "dice_roll",
            "player": player,
            "dice": (text.get("firstDice", 0), text.get("secondDice", 0)),
            "total": text.get("firstDice", 0) + text.get("secondDice", 0),
        }
    elif ltype == 4:  # build settlement or city
        piece = PIECE_NAMES.get(text.get("pieceEnum"), "?")
        return {"type": f"build_{piece}", "player": player}
    elif ltype == 5:  # build road
        return {"type": "build_road", "player": player}
    elif ltype == 11:  # robber placed
        tile_info = text.get("tileInfo", {})
        return {
            "type": "place_robber",
            "player": player,
            "tile": state["robber_tile"],
            "blocked_resource": RESOURCE_NAMES.get(tile_info.get("resourceType")),
        }
    elif ltype == 14:  # steal from player
        stolen = [RESOURCE_NAMES.get(c, f"?{c}") for c in text.get("cardEnums", [])]
        return {"type": "steal", "player": player, "stolen": stolen}
    elif ltype == 20:  # play dev card
        card = DEV_CARD_NAMES.get(text.get("cardEnum"), f"dev_{text.get('cardEnum')}")
        return {"type": "play_dev_card", "player": player, "card": card}
    elif ltype == 44:  # end turn — no playerColor in text, use current state player
        return {"type": "end_turn", "player": state["current_player"]}
    elif ltype == 47:  # receive resources from roll
        cards = [RESOURCE_NAMES.get(c, f"?{c}") for c in text.get("cardsToBroadcast", [])]
        return {"type": "receive_resources", "player": player, "resources": cards}
    elif ltype == 55:  # receive multiple cards (year of plenty)
        cards = [RESOURCE_NAMES.get(c, f"?{c}") for c in text.get("cardEnums", [])]
        return {"type": "receive_cards", "player": player, "resources": cards}
    elif ltype == 60:  # discard
        return {"type": "discard", "player": player}
    elif ltype == 86:  # monopoly steal
        return {
            "type": "monopoly",
            "player": player,
            "resource": RESOURCE_NAMES.get(text.get("cardEnum")),
            "amount": text.get("amountStolen", 0),
        }
    elif ltype == 116:  # trade
        give = [RESOURCE_NAMES.get(c, f"?{c}") for c in text.get("givenCardEnums", [])]
        recv = [RESOURCE_NAMES.get(c, f"?{c}") for c in text.get("receivedCardEnums", [])]
        return {"type": "trade", "player": player, "give": give, "receive": recv}
    elif ltype == 45:  # game over
        return {"type": "game_over", "player": player}
    return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_game(gamedata: dict) -> dict:
    d = gamedata["data"]
    play_order = d["playOrder"]
    players = {p["selectedColor"]: p["username"] for p in d["playerUserStates"]}
    events = d["eventHistory"]["events"]

    state = empty_state(play_order)
    # Init VP and resources for each player
    for color in play_order:
        state["player_vp"][color] = 0
        state["player_resources"][color] = []
    state["player_vp_breakdown"] = {color: {} for color in play_order}

    timeline = []   # list of {turn, player, actions, state_snapshot}
    all_actions = []

    for i, event in enumerate(events):
        state_before = deepcopy(state)
        actions = apply_event(state, event)

        for act in actions:
            if act["type"] in ("game_init", "connected"):
                continue
            record = {
                "event_idx": i,
                "turn": state["turn"],
                "player": act["player"],
                "player_name": players.get(act["player"], f"p{act['player']}"),
                "action": act,
                "resources_before": list(state_before["player_resources"].get(act["player"], [])),
                "resources_after": list(state["player_resources"].get(act["player"], [])),
                "vp": state["player_vp"].get(act["player"], 0),
                "buildings": {
                    "settlements": [
                        vid for vid, cd in state["corners"].items()
                        if cd.get("owner") == act["player"] and cd.get("buildingType") == 1
                    ],
                    "cities": [
                        vid for vid, cd in state["corners"].items()
                        if cd.get("owner") == act["player"] and cd.get("buildingType") == 2
                    ],
                    "roads": [
                        eid for eid, ed in state["edges"].items()
                        if ed.get("owner") == act["player"]
                    ],
                },
                "robber_tile": state["robber_tile"],
            }
            all_actions.append(record)

    return {
        "game_id": d["databaseGameId"],
        "players": players,
        "play_order": play_order,
        "settings": d["gameSettings"],
        "total_events": len(events),
        "total_actions": len(all_actions),
        "actions": all_actions,
        "final_vp": state["player_vp"],
        "final_buildings": {
            str(color): {
                "settlements": [v for v, cd in state["corners"].items() if cd.get("owner") == color and cd.get("buildingType") == 1],
                "cities":      [v for v, cd in state["corners"].items() if cd.get("owner") == color and cd.get("buildingType") == 2],
                "roads":       [e for e, ed in state["edges"].items() if ed.get("owner") == color],
            }
            for color in play_order
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", help="Path to replay gamedata JSON")
    p.add_argument("--out", default=None, help="Output actions JSON path")
    cfg = p.parse_args()

    gamedata = json.loads(Path(cfg.input).read_text())
    result = parse_game(gamedata)

    # Print summary
    print(f"Game {result['game_id']}")
    print(f"Players: {result['players']}")
    print(f"Play order: {result['play_order']}")
    print(f"Total events: {result['total_events']}  |  Actions extracted: {result['total_actions']}")
    print(f"Final VP: {result['final_vp']}")
    print(f"Final buildings: {result['final_buildings']}")

    print("\n--- Action timeline ---")
    prev_turn = -1
    for rec in result["actions"]:
        if rec["turn"] != prev_turn:
            print(f"\n  Turn {rec['turn']}:")
            prev_turn = rec["turn"]
        act = rec["action"]
        res_before = [RESOURCE_NAMES.get(r, r) for r in rec["resources_before"]]
        res_after  = [RESOURCE_NAMES.get(r, r) for r in rec["resources_after"]]
        res_str = f"  [{','.join(res_before)}->{','.join(res_after)}]" if res_before != res_after else ""
        print(f"    {rec['player_name']:15s}  {act['type']:20s}  {str(act.get('dice') or act.get('resources') or act.get('give') or ''):<25}{res_str}  VP={rec['vp']}")

    # Save
    out_path = Path(cfg.out) if cfg.out else Path(cfg.input).with_suffix("").with_name(
        Path(cfg.input).stem.replace("_gamedata", "_actions") + ".json"
    )
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved {result['total_actions']} action records -> {out_path}")


if __name__ == "__main__":
    main()
