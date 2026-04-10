from __future__ import annotations

import base64
import json
import time
from pathlib import Path

from flask import Flask, Response, abort, jsonify, redirect, send_from_directory, stream_with_context


ROOT = Path(__file__).resolve().parent
DASHBOARD_DIR = ROOT / "dashboard"
LIVE_DIR = ROOT / "artifacts" / "live"
STREAM_FRAME_PATH = LIVE_DIR / "preview_live.jpg"


def read_json(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


app = Flask(__name__, static_folder=None)


@app.get("/")
def root():
    return redirect("/dashboard/")


@app.get("/dashboard/")
def dashboard_index():
    return send_from_directory(DASHBOARD_DIR, "index.html")


@app.get("/dashboard/<path:filename>")
def dashboard_assets(filename: str):
    return send_from_directory(DASHBOARD_DIR, filename)


@app.get("/artifacts/live/<path:filename>")
def live_assets(filename: str):
    target = LIVE_DIR / filename
    if not target.exists():
        abort(404)
    return send_from_directory(LIVE_DIR, filename)


@app.get("/api/status")
def api_status():
    payload = read_json(LIVE_DIR / "status.json")
    if payload is None:
        return jsonify({"phase": "idle", "message": "training has not started yet"}), 200
    preview_gif = LIVE_DIR / "preview.gif"
    preview_png = LIVE_DIR / "preview.png"
    preview_file = preview_gif if preview_gif.exists() else preview_png if preview_png.exists() else None
    if preview_file is not None:
        payload["last_preview_path"] = preview_file.name
        payload["preview_updated_at"] = int(preview_file.stat().st_mtime_ns)
    if STREAM_FRAME_PATH.exists():
        payload["stream_path"] = STREAM_FRAME_PATH.name
        payload["stream_updated_at"] = int(STREAM_FRAME_PATH.stat().st_mtime_ns)
    return jsonify(payload)


@app.get("/api/history")
def api_history():
    payload = read_json(LIVE_DIR / "history.json")
    return jsonify(payload or [])


@app.get("/api/logs")
def api_logs():
    log_path = LIVE_DIR / "train.log"
    if not log_path.exists():
        return jsonify({"lines": []})
    text = log_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()[-120:]
    return jsonify({"lines": lines})


@app.get("/api/frame-status")
def api_frame_status():
    if not STREAM_FRAME_PATH.exists():
        return jsonify({"available": False})
    return jsonify(
        {
            "available": True,
            "path": STREAM_FRAME_PATH.name,
            "updated_at": int(STREAM_FRAME_PATH.stat().st_mtime_ns),
        }
    )


@app.get("/api/frame-data")
def api_frame_data():
    if not STREAM_FRAME_PATH.exists():
        return jsonify({"available": False})
    payload = base64.b64encode(STREAM_FRAME_PATH.read_bytes()).decode("ascii")
    return jsonify(
        {
            "available": True,
            "updated_at": int(STREAM_FRAME_PATH.stat().st_mtime_ns),
            "mime": "image/jpeg",
            "data": payload,
        }
    )


@app.get("/api/preview-stream")
def api_preview_stream():
    def generate():
        last_mtime = None
        while True:
            if STREAM_FRAME_PATH.exists():
                mtime = STREAM_FRAME_PATH.stat().st_mtime_ns
                if mtime != last_mtime:
                    payload = STREAM_FRAME_PATH.read_bytes()
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
                    last_mtime = mtime
            time.sleep(0.08)

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)
