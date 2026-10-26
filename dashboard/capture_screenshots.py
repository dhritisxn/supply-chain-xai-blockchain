"""
capture_screenshots.py

Takes full-page screenshots of every dashboard view by driving headless
Microsoft Edge / Chrome over the DevTools protocol (no Selenium/Playwright).
Buttons are clicked exactly as a user would, so the screenshots show real
pipeline and on-chain output.

Prerequisites (for the ledger / approval views to show data):
    ganache --port 8545
    python blockchain/scripts/deploy.py
    python pipeline.py --scan --scenario below_rop      # logs decisions on-chain
    streamlit run dashboard/app.py --server.headless true --server.port 8501

Run:
    python dashboard/capture_screenshots.py [--url http://localhost:8501] [--out dashboard_screenshots]
"""

from __future__ import annotations

import argparse
import base64
import json
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WIDTH, MIN_HEIGHT = 1600, 1000
BROWSERS = [r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Google\Chrome\Application\chrome.exe"]


class Page:
    def __init__(self, ws_url: str):
        self.ws = connect(ws_url, max_size=None, open_timeout=30)
        self._id = 0

    def call(self, method: str, **params):
        self._id += 1
        self.ws.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self._id:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expr: str):
        res = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        return res.get("result", {}).get("value")

    def wait(self, cond: str, timeout: float, what: str):
        end = time.time() + timeout
        while time.time() < end:
            if self.js(cond):
                return
            time.sleep(0.5)
        raise TimeoutError(f"timed out waiting for {what}")

    def idle(self, timeout: float = 120):
        """Wait until Streamlit has finished the current script run."""
        # while a run is in progress the header status widget shows "Running..." (older
        # Streamlit) or a running icon plus a "Stop" button (newer releases)
        busy = ("!!document.querySelector('[data-testid=\"stSpinner\"]') || "
                "[...document.querySelectorAll('[data-testid=\"stStatusWidget\"], "
                "[data-testid=\"stHeader\"] button, header button')]"
                ".some(e => /running|^\\s*stop\\s*$/i.test(e.innerText))")
        time.sleep(1.0)
        end = time.time() + timeout
        quiet_since = None
        while time.time() < end:
            if self.js(busy):
                quiet_since = None
            elif quiet_since is None:
                quiet_since = time.time()
            elif time.time() - quiet_since > 2.0:
                return
            time.sleep(0.5)
        raise TimeoutError("Streamlit did not become idle")

    def click(self, selector: str, text: str):
        ok = self.js(f"""(() => {{
            const el = [...document.querySelectorAll({json.dumps(selector)})]
                .find(e => e.innerText.trim() === {json.dumps(text)});
            if (el) {{ el.click(); return true; }} return false; }})()""")
        if not ok:
            raise RuntimeError(f"no {selector} with text {text!r}")

    def shot(self, path: Path):
        h = self.js("""Math.max(...['[data-testid="stMain"]', '[data-testid="stAppViewContainer"]',
                       'section.main'].map(s => document.querySelector(s)?.scrollHeight || 0),
                       document.body.scrollHeight)""") or MIN_HEIGHT
        self.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=max(MIN_HEIGHT, int(h) + 40),
                  deviceScaleFactor=1, mobile=False)
        time.sleep(1.5)
        data = self.call("Page.captureScreenshot", format="png")["data"]
        path.write_bytes(base64.b64decode(data))
        self.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=MIN_HEIGHT,
                  deviceScaleFactor=1, mobile=False)
        print(f"  saved {path.name}")


def nav(page: Page, view: str):
    page.click('[data-testid="stSidebar"] label', view)
    page.idle()


def forecast_view(page: Page, out: Path):
    print("Forecast & Explainability")
    nav(page, "Forecast & Explainability")
    page.click("button", "Run Forecast")
    page.wait("!!document.querySelector('[data-testid=\"stMetric\"]') || "
              "!!document.querySelector('[data-testid=\"stAlert\"]')", 600, "forecast result")
    page.idle(600)
    page.shot(out / "01_forecast_explainability.png")


def gate_view(page: Page, out: Path):
    print("Confidence Gate (full scan: every model x series, can take several minutes)")
    nav(page, "Confidence Gate")
    page.click("button", "Run Full Scan (18 combinations)")
    page.wait("!!document.querySelector('[data-testid=\"stDataFrame\"]') || "
              "!!document.querySelector('[data-testid=\"stException\"]')", 3600, "scan result")
    page.idle(600)
    page.shot(out / "02_confidence_gate_full_scan.png")
    page.click("button", "Evaluate")
    page.wait("!!document.querySelector('[data-testid=\"stJson\"]')", 600, "single evaluation")
    page.idle(600)
    page.shot(out / "03_confidence_gate_single_evaluation.png")


def ledger_view(page: Page, out: Path):
    print("Audit Ledger")
    nav(page, "Audit Ledger")
    page.shot(out / "04_audit_ledger_decisions.png")
    page.click('[role="tab"]', "Purchase Orders")
    time.sleep(1.5)
    page.shot(out / "05_audit_ledger_purchase_orders.png")
    page.click('[role="tab"]', "Integrity Check")
    time.sleep(1.0)
    page.click("button", "Run Integrity Check")
    page.idle(300)
    page.click('[role="tab"]', "Integrity Check")  # tab can reset on rerun
    time.sleep(1.5)
    page.shot(out / "06_audit_ledger_integrity_check.png")


def approval_view(page: Page, out: Path):
    print("Approval Queue")
    nav(page, "Approval Queue")
    page.js("[...document.querySelectorAll('[data-testid=\"stExpander\"] summary')].slice(0, 2)"
            ".forEach(s => s.click())")
    time.sleep(1.5)
    page.shot(out / "07_approval_queue.png")


def status_view(page: Page, out: Path):
    print("System Status")
    nav(page, "System Status")
    page.shot(out / "08_system_status.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8501")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "dashboard_screenshots")
    ap.add_argument("--port", type=int, default=9333)
    ap.add_argument("--from-view", type=int, default=1, choices=range(1, 6),
                    help="resume at view N (1 forecast, 2 gate, 3 ledger, 4 approvals, 5 status)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    exe = next((b for b in BROWSERS if Path(b).exists()), None) or shutil.which("msedge") or shutil.which("chrome")
    if exe is None:
        raise SystemExit("Edge or Chrome is required")
    profile = tempfile.mkdtemp(prefix="dash_shots_")
    browser = subprocess.Popen([exe, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                                f"--remote-debugging-port={args.port}", f"--user-data-dir={profile}",
                                f"--window-size={WIDTH},{MIN_HEIGHT}", "about:blank"],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(60):
            try:
                targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{args.port}/json"))
                ws_url = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise SystemExit("browser DevTools endpoint did not come up")

        page = Page(ws_url)
        page.call("Page.enable")
        page.call("Emulation.setDeviceMetricsOverride", width=WIDTH, height=MIN_HEIGHT,
                  deviceScaleFactor=1, mobile=False)
        page.call("Page.navigate", url=args.url)
        page.wait("!!document.querySelector('[data-testid=\"stSidebar\"] label')", 90, "dashboard to load")
        page.idle()
        views = [forecast_view, gate_view, ledger_view, approval_view, status_view]
        for view in views[args.from_view - 1:]:
            view(page, args.out)
    finally:
        browser.terminate()
        try:
            browser.wait(10)
        except subprocess.TimeoutExpired:
            browser.kill()
        shutil.rmtree(profile, ignore_errors=True)


if __name__ == "__main__":
    main()
