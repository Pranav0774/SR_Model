"""Start the app and a public https link to it. Press Ctrl+C (or close the window) to stop both.

    python share.py

Uses a free Cloudflare "quick tunnel": no account is needed and the link changes every time.
The first run downloads cloudflared (about 55 MB) from Cloudflare's official GitHub release and
checks it against the SHA-256 checksum GitHub publishes for that file.
"""
import ctypes
import hashlib
import json
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOL = ROOT / "tools" / "cloudflared.exe"
RELEASE_API = "https://api.github.com/repos/cloudflare/cloudflared/releases/latest"
ASSET = "cloudflared-windows-amd64.exe"
PORT = 5000
LOCAL = f"http://127.0.0.1:{PORT}"


def get(url, timeout=10):
    req = urllib.request.Request(url, headers={"User-Agent": "share.py"})
    return urllib.request.urlopen(req, timeout=timeout)


def download_cloudflared():
    print("cloudflared is not here yet. Downloading it from Cloudflare's official release (about 55 MB) ...")
    release = json.load(get(RELEASE_API))
    asset = next((a for a in release["assets"] if a["name"] == ASSET), None)
    if asset is None or not (asset.get("digest") or "").startswith("sha256:"):
        sys.exit("Could not find the file or its checksum in the release, so nothing was downloaded.")
    expected = asset["digest"].split(":", 1)[1]
    TOOL.parent.mkdir(exist_ok=True)
    tmp = TOOL.with_suffix(".part")
    sha = hashlib.sha256()
    with get(asset["browser_download_url"], timeout=60) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            sha.update(chunk)
            f.write(chunk)
    if sha.hexdigest() != expected:
        tmp.unlink()
        sys.exit("The downloaded file did not match Cloudflare's checksum, so it was deleted.")
    tmp.replace(TOOL)
    print(f"Downloaded and verified cloudflared {release['tag_name']}.")


def app_status():
    try:
        return json.load(get(LOCAL + "/api/status", timeout=3))
    except Exception:
        return None


def main():
    if not TOOL.exists():
        download_cloudflared()

    # keep Windows from going to sleep while we are sharing (closing the lid can still sleep it)
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)

    procs = []
    try:
        if app_status() is None:
            print("Starting the app ...")
            log = open(ROOT / "share_app.log", "w")
            procs.append(subprocess.Popen([sys.executable, "app.py"], cwd=ROOT, stdout=log, stderr=log))
            for _ in range(60):
                if app_status() is not None:
                    break
                if procs[0].poll() is not None:
                    sys.exit("The app stopped straight away. See share_app.log.")
                time.sleep(1)
            else:
                sys.exit("The app did not start. See share_app.log.")
        else:
            print("The app is already running, so I will reuse it.")

        print("Opening the tunnel ...")
        tunnel = subprocess.Popen([str(TOOL), "tunnel", "--url", LOCAL, "--no-autoupdate"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, encoding="utf-8")
        procs.append(tunnel)

        link = None
        tlog = open(ROOT / "share_tunnel.log", "w", encoding="utf-8")
        for line in tunnel.stderr:
            tlog.write(line)
            tlog.flush()
            m = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", line)
            if m:
                link = m.group(0)
                break
        if link is None:
            sys.exit("Could not get a link from Cloudflare. See share_tunnel.log.")

        # keep draining the tunnel's output so it never blocks
        threading.Thread(target=lambda: [tlog.write(x) or tlog.flush() for x in tunnel.stderr], daemon=True).start()

        subprocess.run("clip", input=link, text=True, shell=True)
        print("\n" + "=" * 62)
        print("  Send this link to your friend (it is also copied to your clipboard):\n")
        print("  " + link)
        print("\n  Keep this window open. Press Ctrl+C or close it to stop sharing.")
        print("=" * 62 + "\n")

        announced = False
        while True:
            if tunnel.poll() is not None:
                print("The tunnel stopped. Run share.bat again for a new link.")
                break
            if procs[0] is not tunnel and procs[0].poll() is not None:
                print("The app stopped. Run share.bat again.")
                break
            st = app_status()
            if st and st.get("ready") and not announced:
                print("Models are loaded. Everything is ready to use.")
                announced = True
            time.sleep(3)
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    main()
