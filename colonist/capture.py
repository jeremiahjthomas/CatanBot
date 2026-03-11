"""
capture.py

Download a Colonist.io replay by game ID using a real browser (Playwright).
Playwright handles Cloudflare Turnstile tokens automatically.

Usage:
    python colonist/capture.py 214191014
    python colonist/capture.py 214191014 214258421 214260368

Output:
    colonist/replays/{game_id}.json

One-time setup:
    pip install playwright
    playwright install chromium

Cookie setup (one-time, needed so you're logged in as premium):
    1. Export cookies from your browser via Cookie-Editor or DevTools
    2. Paste the JSON array into: colonist/colonist_cookies.json
    -- OR --
    Set COLONIST_USER / COLONIST_PASS env vars and we'll log in headlessly.
"""

import sys
import json
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout

REPLAYS_DIR = Path(__file__).parent / "replays"
COOKIE_FILE = Path(__file__).parent / "colonist_cookies.json"
BASE_URL    = "https://colonist.io"
TIMEOUT_MS  = 30_000   # 30 s to wait for the API response


def _load_cookies() -> list[dict] | None:
    if not COOKIE_FILE.exists():
        return None
    raw = json.loads(COOKIE_FILE.read_text())
    # Normalise to Playwright cookie format
    pw_cookies = []
    for c in raw:
        pw_c = {
            "name":   c["name"],
            "value":  c["value"],
            "domain": c.get("domain", ".colonist.io"),
            "path":   c.get("path", "/"),
        }
        if "sameSite" in c and c["sameSite"] in ("Strict", "Lax", "None"):
            pw_c["sameSite"] = c["sameSite"]
        if "httpOnly" in c:
            pw_c["httpOnly"] = bool(c["httpOnly"])
        if "secure" in c:
            pw_c["secure"] = bool(c["secure"])
        if "expirationDate" in c:
            pw_c["expires"] = int(c["expirationDate"])
        pw_cookies.append(pw_c)
    return pw_cookies


def download(game_id: str, context, out_dir: Path = REPLAYS_DIR) -> dict:
    """Open a browser page, navigate to the replay, intercept the data response."""
    out_path = out_dir / f"{game_id}.json"
    if out_path.exists():
        print(f"  {game_id}: already exists, skipping")
        return json.loads(out_path.read_text())

    page = context.new_page()
    captured: dict | None = None

    def handle_response(response):
        nonlocal captured
        if "replay" in response.url and response.status != 200:
            print(f"    [{response.status}] {response.url}")
        if "data-from-game-id" in response.url and str(game_id) in response.url:
            try:
                body = response.text()
            except Exception:
                body = "<unreadable>"
            print(f"    data-from-game-id: {response.status}  body={body[:200]}")
            if response.status == 200:
                try:
                    captured = response.json()
                except Exception:
                    pass

    page.on("response", handle_response)

    try:
        # Try color=1 first, then color=5
        for color in [1, 5]:
            captured = None
            url = f"{BASE_URL}/replay?gameId={game_id}&playerColor={color}"
            page.goto(url, wait_until="domcontentloaded", timeout=TIMEOUT_MS)
            # Wait up to 30 s for the intercept to fire
            deadline = time.time() + 30
            while captured is None and time.time() < deadline:
                page.wait_for_timeout(300)
            if captured is not None:
                break

        if captured is None:
            raise RuntimeError(
                f"No data-from-game-id response captured for {game_id}. "
                "Make sure you're logged in as a premium account."
            )

        events = captured.get("data", {}).get("eventHistory", {}).get("events", [])
        out_dir.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(captured, indent=2))
        print(f"  {game_id}: saved {len(events)} events -> {out_path.name}")
        return captured

    finally:
        page.close()


def login_and_save_cookies():
    """Open Chrome with your real profile, navigate to colonist.io, save cookies."""
    import os
    user_data_dir = os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\User Data")
    print(f"Using Chrome profile: {user_data_dir}")
    print("NOTE: close all Chrome windows before running this, or it may fail.")
    print("Opening colonist.io — make sure you're logged in, then press Enter here.")
    with sync_playwright() as pw:
        context = pw.chromium.launch_persistent_context(
            user_data_dir,
            channel="chrome",
            headless=False,
        )
        page = context.new_page()
        page.goto(f"{BASE_URL}/", timeout=30_000)
        input("  [Press Enter once you are logged in to colonist.io] ")
        print("  Enter received — saving cookies...")
        cookies = context.cookies()
        COOKIE_FILE.parent.mkdir(exist_ok=True)
        COOKIE_FILE.write_text(json.dumps(cookies, indent=2))
        print(f"  Saved {len(cookies)} cookies -> {COOKIE_FILE}")
        context.close()


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage:")
        print("  python colonist/capture.py --login               # save cookies interactively")
        print("  python colonist/capture.py <game_id> ...         # download replays")
        print("  python colonist/capture.py --dir <dir> <id> ...  # download to custom folder")
        sys.exit(0 if "--help" in sys.argv or "-h" in sys.argv else 1)

    if sys.argv[1] == "--login":
        login_and_save_cookies()
        return

    args = sys.argv[1:]
    out_dir = REPLAYS_DIR
    if args[0] == "--dir":
        out_dir = Path(args[1])
        args = args[2:]
    game_ids = args
    cookies  = _load_cookies()

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=False,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context()
        # Hide webdriver flag so Turnstile doesn't detect automation
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        if cookies:
            context.add_cookies(cookies)
        else:
            print("WARNING: no colonist_cookies.json found — run with --login first")

        for i, gid in enumerate(game_ids):
            try:
                download(gid, context, out_dir)
            except Exception as e:
                print(f"  {gid}: ERROR — {e}")
            if i < len(game_ids) - 1:
                time.sleep(1)

        context.close()
        browser.close()


if __name__ == "__main__":
    main()
