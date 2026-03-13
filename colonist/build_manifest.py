"""
build_manifest.py

Scans colonist/replays/*.json, parses each game, and writes a manifest to
colonist/replays/manifest.jsonl — one JSON record per replay.

Usage:
    python colonist/build_manifest.py
    python colonist/build_manifest.py --replays-dir colonist/replays --out colonist/replays/manifest.jsonl
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Allow importing parse_replay from the same package directory
sys.path.insert(0, str(Path(__file__).parent))
from parse_replay import parse_game, LOG_TYPE_NAMES

PARSER_VERSION = "1.0"

# Turn-count thresholds for game_length_bucket
SHORT_MAX  = 30
MEDIUM_MAX = 70  # above this → "long"

# Settings keys that define the ruleset fingerprint
def _ruleset_string(settings: dict) -> str:
    vp   = settings.get("victoryPointsToWin", "?")
    dice = "balanced" if settings.get("diceSetting", 0) == 0 else f"dice_{settings.get('diceSetting')}"
    fr   = "friendly_robber" if settings.get("friendlyRobber") else "standard_robber"
    dl   = f"discard{settings.get('cardDiscardLimit', '?')}"
    mp   = f"{settings.get('maxPlayers', '?')}p"
    return f"{mp}_{vp}vp_{dice}_{fr}_{dl}"


def _scan_raw_events(events: list, mapping: dict) -> dict:
    """
    Single pass over raw events to collect validation signals:
    - unknown log types
    - edge IDs not in the mapping
    - vertex IDs not in the mapping
    """
    known_ltypes    = set(LOG_TYPE_NAMES.keys())
    mapped_vertices = set(mapping.get("col_to_catan_vertex", {}).keys())
    mapped_edges    = set(mapping.get("col_to_catan_edge", {}).keys())

    unknown_ltypes   = set()
    unknown_edges    = set()
    unknown_vertices = set()

    for ev in events:
        sc = ev.get("stateChange", {})

        # Check vertex IDs seen on the board
        for vid in sc.get("mapState", {}).get("tileCornerStates", {}):
            if str(vid) not in mapped_vertices:
                unknown_vertices.add(str(vid))

        # Check edge IDs seen on the board
        for eid in sc.get("mapState", {}).get("tileEdgeStates", {}):
            if str(eid) not in mapped_edges:
                unknown_edges.add(str(eid))

        # Check log types
        gls = sc.get("gameLogState", {})
        if isinstance(gls, dict):
            for entry in gls.values():
                if isinstance(entry, dict):
                    text = entry.get("text", {})
                    if isinstance(text, dict):
                        lt = text.get("type")
                        if lt is not None and lt not in known_ltypes:
                            unknown_ltypes.add(lt)

    return {
        "unknown_ltypes":   sorted(unknown_ltypes),
        "unknown_edges":    sorted(unknown_edges),
        "unknown_vertices": sorted(unknown_vertices),
    }


def build_entry(replay_path: Path, mapping: dict) -> dict:
    """Parse one replay file and return its manifest entry dict."""
    entry = {
        "game_id":              None,
        "file_path":            str(replay_path),
        "download_timestamp":   datetime.fromtimestamp(
                                    replay_path.stat().st_mtime, tz=timezone.utc
                                ).isoformat(),
        "parser_version":       PARSER_VERSION,
        "colonist_ruleset":     None,
        "players":              [],
        "winner":               None,
        "final_vp":             {},
        "turn_count":           None,
        "action_count":         None,
        "decision_action_count": None,
        "parse_success":        False,
        "parse_warnings":       [],
        "mapping_coverage_ok":  None,
        "has_unknown_events":   None,
        "has_unknown_edges":    None,
        "has_unknown_vertices": None,
        "unknown_ltypes":       [],
        "unknown_edges":        [],
        "unknown_vertices":     [],
        "game_length_bucket":   None,
        "is_2p":                None,   # True only for 2-player games
        "training_eligible":    None,   # True if 2p + parse_success
        "notes":                "",
    }

    # Non-decision actions — present in records but not useful as training targets
    NON_DECISION = {"dice_roll", "receive_resources", "receive_cards",
                    "end_turn", "game_over", "game_init", "connected"}

    try:
        raw = json.loads(replay_path.read_text(encoding="utf-8"))
    except Exception as e:
        entry["parse_warnings"].append(f"json_load_error: {e}")
        return entry

    try:
        d      = raw["data"]
        events = d["eventHistory"]["events"]
        entry["game_id"] = str(d.get("databaseGameId", replay_path.stem))

        # Ruleset
        settings = d.get("gameSettings", {})
        entry["colonist_ruleset"] = _ruleset_string(settings)

        # Players
        player_states = d.get("playerUserStates", [])
        if isinstance(player_states, dict):
            player_states = list(player_states.values())
        entry["players"] = [
            {"color": p["selectedColor"], "name": p["username"]}
            for p in player_states
        ]

        # Scan raw events for unknown IDs / ltypes
        scan = _scan_raw_events(events, mapping)
        entry["unknown_ltypes"]   = scan["unknown_ltypes"]
        entry["unknown_edges"]    = scan["unknown_edges"]
        entry["unknown_vertices"] = scan["unknown_vertices"]
        entry["has_unknown_events"]   = len(scan["unknown_ltypes"]) > 0
        entry["has_unknown_edges"]    = len(scan["unknown_edges"]) > 0
        entry["has_unknown_vertices"] = len(scan["unknown_vertices"]) > 0
        entry["mapping_coverage_ok"]  = not (entry["has_unknown_edges"] or entry["has_unknown_vertices"])

        if scan["unknown_ltypes"]:
            entry["parse_warnings"].append(f"unknown_log_types: {scan['unknown_ltypes']}")
        if scan["unknown_edges"]:
            entry["parse_warnings"].append(f"unmapped_edges: {scan['unknown_edges']}")
        if scan["unknown_vertices"]:
            entry["parse_warnings"].append(f"unmapped_vertices: {scan['unknown_vertices']}")

        # Full parse
        result = parse_game(raw)
        entry["parse_success"] = True

        entry["action_count"] = result["total_actions"]
        entry["final_vp"]     = {str(k): v for k, v in result["final_vp"].items()}

        # Turn count: max turn number seen in action records
        if result["actions"]:
            entry["turn_count"] = max(rec["turn"] for rec in result["actions"])
        else:
            entry["turn_count"] = 0

        # Decision action count (training-relevant)
        entry["decision_action_count"] = sum(
            1 for rec in result["actions"]
            if rec["action"]["type"] not in NON_DECISION
        )

        # Winner: player from the game_over action
        winner = None
        for rec in result["actions"]:
            if rec["action"]["type"] == "game_over":
                winner = rec["action"].get("player")
                break
        # Fallback: highest VP
        if winner is None and result["final_vp"]:
            winner_str = max(result["final_vp"], key=lambda c: result["final_vp"][c])
            winner = int(winner_str)
            entry["parse_warnings"].append("winner_inferred_from_vp")
        entry["winner"] = winner

        # Game length bucket
        tc = entry["turn_count"]
        if tc is None:
            entry["game_length_bucket"] = "unknown"
        elif tc <= SHORT_MAX:
            entry["game_length_bucket"] = "short"
        elif tc <= MEDIUM_MAX:
            entry["game_length_bucket"] = "medium"
        else:
            entry["game_length_bucket"] = "long"

        # 2p flag and training eligibility
        entry["is_2p"] = settings.get("maxPlayers", 0) == 2
        entry["training_eligible"] = entry["is_2p"] and entry["parse_success"]

        # Warn if not a 2p game
        if not entry["is_2p"]:
            entry["parse_warnings"].append(f"not_2p_game (maxPlayers={settings.get('maxPlayers')})")

        # Warn if no game_over event (truncated replay?)
        if winner is None:
            entry["parse_warnings"].append("no_game_over_event")

    except Exception:
        entry["parse_success"] = False
        entry["parse_warnings"].append(f"parse_error: {traceback.format_exc(limit=3)}")

    return entry


def build_manifest(replays_dir: Path, out_path: Path, mapping_path: Path) -> None:
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))

    replay_files = sorted(replays_dir.glob("*.json"))
    # Skip the manifest itself if it ends up in the same folder
    replay_files = [f for f in replay_files if f.name != out_path.name]

    if not replay_files:
        print(f"No replay JSON files found in {replays_dir}")
        return

    entries = []
    for replay_path in replay_files:
        print(f"  processing {replay_path.name} ...", end=" ", flush=True)
        entry = build_entry(replay_path, mapping)
        entries.append(entry)
        status = "OK" if entry["parse_success"] else "FAIL"
        warnings = f"  [{len(entry['parse_warnings'])} warnings]" if entry["parse_warnings"] else ""
        print(f"{status}{warnings}")

    out_path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8"
    )

    # Summary
    n_ok        = sum(1 for e in entries if e["parse_success"])
    n_warn      = sum(1 for e in entries if e["parse_warnings"])
    n_unmapped  = sum(1 for e in entries if not e["mapping_coverage_ok"])
    n_2p        = sum(1 for e in entries if e["is_2p"])
    n_eligible  = sum(1 for e in entries if e["training_eligible"])
    total_decisions     = sum(e["decision_action_count"] or 0 for e in entries)
    eligible_decisions  = sum(e["decision_action_count"] or 0 for e in entries if e["training_eligible"])

    print(f"\nManifest written to {out_path}")
    print(f"  {len(entries)} replays  |  {n_ok} parsed OK  |  {n_2p} are 2p  |  {n_eligible} training-eligible")
    print(f"  {n_warn} with warnings  |  {n_unmapped} with unmapped IDs")
    print(f"  Decision actions — all: {total_decisions}  |  training-eligible only: {eligible_decisions}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replays-dir", default="colonist/replays",
                        help="Directory containing replay JSON files")
    parser.add_argument("--out", default="colonist/manifest.jsonl",
                        help="Output manifest path")
    parser.add_argument("--mapping", default="colonist/colonist_to_catan_mapping.json",
                        help="Colonist coordinate mapping file")
    args = parser.parse_args()

    replays_dir  = Path(args.replays_dir)
    out_path     = Path(args.out)
    mapping_path = Path(args.mapping)

    if not replays_dir.exists():
        print(f"Replays directory not found: {replays_dir}")
        sys.exit(1)
    if not mapping_path.exists():
        print(f"Mapping file not found: {mapping_path}")
        sys.exit(1)

    print(f"Scanning {replays_dir} ...")
    build_manifest(replays_dir, out_path, mapping_path)


if __name__ == "__main__":
    main()
