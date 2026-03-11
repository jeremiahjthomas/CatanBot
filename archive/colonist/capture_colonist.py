"""
capture_colonist.py

Captures all WebSocket messages from a Colonist.io replay page,
decodes them from MessagePack, and saves to JSON.

Usage:
    python capture_colonist.py                          # game 207429263 (default)
    python capture_colonist.py 207429263
    python capture_colonist.py 207429263 --wait 10
    python capture_colonist.py 207429263 --login        # pause for login before navigating
"""

import asyncio
import argparse
import base64
import json
import os
import msgpack
from pathlib import Path
from playwright.async_api import async_playwright

COOKIE_FILE = Path(__file__).parent / "colonist_cookies.json"


INTERCEPT_JS = """
// Hide automation fingerprint
Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

window._wsMessages = [];
const _OrigWS = window.WebSocket;
window.WebSocket = function(...args) {
    const ws = new _OrigWS(...args);
    ws.addEventListener('message', async (e) => {
        try {
            if (e.data instanceof Blob) {
                const buf = await e.data.arrayBuffer();
                window._wsMessages.push({
                    kind: 'binary',
                    data: Array.from(new Uint8Array(buf)),
                    url: ws.url,
                    ts: Date.now()
                });
            } else {
                window._wsMessages.push({
                    kind: 'text',
                    data: e.data,
                    url: ws.url,
                    ts: Date.now()
                });
            }
        } catch(err) {}
    });
    return ws;
};
"""


def decode_message(msg: dict) -> dict:
    """Decode a captured WebSocket message from MessagePack or text."""
    if msg["kind"] == "text":
        try:
            return {"kind": "text", "decoded": json.loads(msg["data"]), "ts": msg["ts"]}
        except Exception:
            return {"kind": "text", "decoded": msg["data"], "ts": msg["ts"]}
    else:
        raw = bytes(msg["data"])
        try:
            decoded = msgpack.unpackb(raw, raw=False)
            return {"kind": "binary", "decoded": decoded, "ts": msg["ts"], "size": len(raw)}
        except Exception as ex:
            return {"kind": "binary_error", "error": str(ex), "size": len(raw), "ts": msg["ts"]}


def get_colonist_cookies() -> list:
    """Load colonist.io cookies from exported JSON file."""
    if not COOKIE_FILE.exists():
        raise FileNotFoundError(
            f"Cookie file not found: {COOKIE_FILE}\n"
            "  1. Install 'Cookie-Editor' extension in Chrome\n"
            "  2. Go to colonist.io while logged in\n"
            "  3. Click Cookie-Editor → Export → Export as JSON → Copy\n"
            f"  4. Paste into: {COOKIE_FILE}"
        )
    raw = json.loads(COOKIE_FILE.read_text())
    # Cookie-Editor exports a list; normalize each entry for Playwright
    cookies = []
    for c in raw:
        cookie = {
            "name":   c["name"],
            "value":  c["value"],
            "domain": c.get("domain", ".colonist.io"),
            "path":   c.get("path", "/"),
            "secure": c.get("secure", False),
        }
        if "expirationDate" in c:
            cookie["expires"] = int(c["expirationDate"])
        cookies.append(cookie)
    print(f"Loaded {len(cookies)} colonist.io cookies from {COOKIE_FILE.name}")
    return cookies


async def capture(game_id: str, wait_seconds: int) -> list:  # noqa: C901
    replay_url = f"https://colonist.io/replay?gameId={game_id}&playerColor=5"

    async with async_playwright() as p:
        # Connect to the already-running Chrome (launched with --remote-debugging-port=9222)
        print("Connecting to Chrome on port 9222...")
        browser = await p.chromium.connect_over_cdp("http://127.0.0.1:9222")

        # Find the colonist replay tab
        page = None
        for ctx in browser.contexts:
            for pg in ctx.pages:
                print(f"  Found tab: {pg.url}")
                if "colonist.io/replay" in pg.url:
                    page = pg
                    break

        if page is None:
            # Open a new tab and navigate
            print(f"No replay tab found — opening {replay_url}")
            ctx = browser.contexts[0]
            page = await ctx.new_page()
            await page.goto(replay_url, wait_until="domcontentloaded", timeout=60000)
            print(f"Current URL: {page.url}")

        print(f"Using tab: {page.url}")

        # Open a CDP session directly on this page target
        cdp = await page.context.new_cdp_session(page)

        # Enable Network domain BEFORE navigating — captures WS frames from page load
        await cdp.send("Network.enable")

        # Inject JS interceptor BEFORE navigating so it runs on page load
        await cdp.send("Page.addScriptToEvaluateOnNewDocument", {"source": INTERCEPT_JS})

        # Intercept HTTP responses
        http_responses = []

        async def on_response(response):
            url = response.url
            if "colonist.io" in url and response.status == 200:
                try:
                    body = await response.body()
                    if len(body) > 200:
                        print(f"  HTTP {response.status} {len(body):6d}b  {url}")
                        entry = {"url": url, "status": response.status, "size": len(body)}
                        if "api/" in url:
                            entry["body"] = body.decode("utf-8", errors="replace")
                            print(f"    *** API response captured ({len(body)} bytes) ***")
                        http_responses.append(entry)
                except Exception:
                    pass

        page.on("response", on_response)

        # Collect CDP-level WebSocket frames — must be registered before navigate
        cdp_frames = []
        cdp.on("Network.webSocketFrameReceived", lambda p: cdp_frames.append(p))

        # Navigate — interceptors are now active before page loads
        print(f"Navigating to: {replay_url}")
        await page.goto(replay_url, wait_until="domcontentloaded", timeout=60000)
        print(f"Page loaded. URL: {page.url}")

        await asyncio.get_event_loop().run_in_executor(
            None, input,
            "\n>>> Start/play the replay in Chrome, then press Enter here...\n"
        )

        print(f"Capturing for {wait_seconds}s...")
        await asyncio.sleep(wait_seconds)

        print(f"\n--- Captured {len(http_responses)} HTTP responses, {len(cdp_frames)} WS frames ---")

        # Save all captured API responses
        replays_dir = Path(__file__).parent / "replays"
        for r in http_responses:
            if "body" not in r:
                continue
            url = r["url"]
            if "api/replay" in url:
                game_path = replays_dir / f"replay_{game_id}_gamedata.json"
                game_path.write_text(r["body"])
                events = json.loads(r["body"])["data"]["eventHistory"]["events"]
                print(f"Game data saved → {game_path}  ({len(events)} events)")
            else:
                # Save other API responses for inspection
                import re
                slug = re.sub(r"[^\w]", "_", url.split("colonist.io/")[-1])[:60]
                out = replays_dir / f"api_{game_id}_{slug}.json"
                out.write_text(r["body"])
                print(f"  API response saved → {out}")

        # Return JS-intercepted messages + CDP WS frames
        js_messages = await page.evaluate("window._wsMessages || []")

        # Save CDP WS frames separately for inspection
        if cdp_frames:
            cdp_path = Path(__file__).parent / "replays" / f"replay_{game_id}_cdp_frames.json"
            cdp_path.write_text(json.dumps(cdp_frames, indent=2, default=str))
            print(f"CDP WS frames saved → {cdp_path}  ({len(cdp_frames)} frames)")

        raw_messages = js_messages

    print(f"Captured {len(raw_messages)} raw messages")
    return raw_messages


def main():
    p = argparse.ArgumentParser()
    p.add_argument("game_id", nargs="?", default="207429263")
    p.add_argument("--wait",     type=int,  default=8,
                   help="Seconds to wait for messages (default 8)")
    cfg = p.parse_args()

    raw = asyncio.run(capture(cfg.game_id, cfg.wait))

    # Decode all messages
    decoded = [decode_message(m) for m in raw]

    # Save full dump
    out_path = Path(__file__).parent / "replays" / f"replay_{cfg.game_id}_messages.json"
    with open(out_path, "w") as f:
        json.dump(decoded, f, indent=2, default=str)
    print(f"Saved {len(decoded)} decoded messages → {out_path}")

    # Print summary of each message type
    print("\n--- Message summary ---")
    for i, msg in enumerate(decoded):
        d = msg.get("decoded", {})
        size = msg.get("size", "?")
        if isinstance(d, dict):
            mtype = d.get("type", d.get("action", "?"))
            keys = list(d.keys())[:5]
            print(f"  [{i:3d}] kind={msg['kind']:8s}  size={size:6}  type={mtype}  keys={keys}")
        else:
            print(f"  [{i:3d}] kind={msg['kind']:8s}  size={size:6}  value={str(d)[:80]}")


if __name__ == "__main__":
    main()
