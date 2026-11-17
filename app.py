"""Web interface backend. Run it from anywhere:  python app.py
Then open http://127.0.0.1:5000 in your browser.

Files:  app.py (this file), robot.py (robot connection, mock for now),
        templates/index.html (the page), templates/report.html (printable report)
Data:   inspections/inspections.db, inspections/images, inspections/results
"""
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
os.chdir(BASE)   # predict_one loads the model from a relative path

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory

import predict_one
import robot

DATA = BASE / "inspections"
IMAGES = DATA / "images"
RESULTS = DATA / "results"
TMP = DATA / "tmp"          # analysed but not yet saved
DB = DATA / "inspections.db"
for folder in (IMAGES, RESULTS, TMP):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024   # 30 MB per upload

LOCK = threading.Lock()                  # one analysis at a time on the model
MODEL = {"state": "loading", "error": ""}
TOKEN_RE = re.compile(r"^[0-9a-f]{12}$")
REVIEW_STATES = ("pending", "accepted", "rejected")
PUBLIC_FOLDERS = ("images", "results", "tmp")   # the only folders the browser may read


# ---------- database (sqlite3 is built into Python) ----------

def run(sql, args=()):
    """Run a write statement and return the new row id."""
    with closing(sqlite3.connect(DB)) as con:
        with con:
            return con.execute(sql, args).lastrowid


def fetch(sql, args=()):
    with closing(sqlite3.connect(DB)) as con:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(sql, args).fetchall()]


def init_db():
    run("""CREATE TABLE IF NOT EXISTS inspections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT, date TEXT, time TEXT, zone TEXT, notes TEXT,
        image_file TEXT, overlay_file TEXT, mask_file TEXT,
        corrosion_ratio REAL, model TEXT,
        review_status TEXT DEFAULT 'pending',
        review_comment TEXT DEFAULT '', reviewed_at TEXT)""")


def present(row):
    """Add the fields the page needs."""
    row["image_url"] = "/files/" + row["image_file"]
    row["overlay_url"] = "/files/" + row["overlay_file"]
    row["corrosion_percent"] = round(100 * row["corrosion_ratio"], 1)
    return row


def model_name():
    return Path(predict_one.CKPT).parent.name


def clean_tmp():
    """Delete analyses that were never saved and are older than a day."""
    cutoff = time.time() - 24 * 3600
    for f in TMP.iterdir():
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink()


def load_model_in_background():
    try:
        with LOCK:
            predict_one.load_model()
        MODEL["state"] = "ready"
    except Exception as e:
        MODEL["state"] = "error"
        MODEL["error"] = str(e)


# ---------- pages and files ----------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/files/<path:name>")
def files(name):
    if name.split("/")[0] not in PUBLIC_FOLDERS:
        abort(404)
    return send_from_directory(DATA, name)


@app.route("/report")
def report():
    where, args = [], []
    row_id = request.args.get("id", type=int)
    zone = request.args.get("zone", "").strip()
    if row_id:
        where.append("id=?")
        args.append(row_id)
    if zone:
        where.append("zone=?")
        args.append(zone)
    sql = "SELECT * FROM inspections" + (" WHERE " + " AND ".join(where) if where else "")
    rows = [present(r) for r in fetch(sql + " ORDER BY id", args)]
    counts = {s: sum(1 for r in rows if r["review_status"] == s) for s in REVIEW_STATES}
    return render_template("report.html", rows=rows, counts=counts, model=model_name(),
                           generated=datetime.now().strftime("%Y-%m-%d %H:%M"))


# ---------- robot ----------

@app.route("/api/status")
def api_status():
    return jsonify({"robot": robot.status(), "model": MODEL})


@app.route("/api/command", methods=["POST"])
def api_command():
    cmd = (request.get_json(silent=True) or {}).get("cmd", "")
    try:
        return jsonify(robot.send_command(cmd))
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        return jsonify({"ok": False, "error": f"Robot error: {e}"}), 500


# ---------- analysis ----------

@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    upload = request.files.get("image")
    if upload is None:
        return jsonify({"error": "No image received."}), 400
    img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "That file is not a readable image (use JPG or PNG)."}), 400
    if MODEL["state"] == "error":
        return jsonify({"error": "The model failed to load: " + MODEL["error"]}), 500

    token = uuid.uuid4().hex[:12]
    src = TMP / f"{token}.jpg"
    cv2.imwrite(str(src), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    t0 = time.perf_counter()
    try:
        with LOCK:
            result = predict_one.analyze_image(src)
    except Exception as e:
        src.unlink(missing_ok=True)
        return jsonify({"error": f"Analysis failed: {e}"}), 500
    seconds = time.perf_counter() - t0

    cv2.imwrite(str(TMP / f"{token}_overlay.jpg"), result["overlay"], [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(TMP / f"{token}_mask.png"), result["mask"])
    return jsonify({
        "token": token,
        "corrosion_ratio": result["corrosion_ratio"],
        "corrosion_percent": round(100 * result["corrosion_ratio"], 1),
        "seconds": round(seconds, 2),
        "image_url": f"/files/tmp/{token}.jpg",
        "overlay_url": f"/files/tmp/{token}_overlay.jpg",
    })


# ---------- inspections ----------

@app.route("/api/inspections", methods=["GET"])
def api_list():
    return jsonify([present(r) for r in fetch("SELECT * FROM inspections ORDER BY id DESC LIMIT 200")])


@app.route("/api/inspections", methods=["POST"])
def api_save():
    body = request.get_json(silent=True) or {}
    token = str(body.get("token", ""))
    zone = str(body.get("zone", "")).strip()[:40]
    notes = str(body.get("notes", "")).strip()[:2000]
    if not TOKEN_RE.match(token):
        return jsonify({"error": "Capture and analyse an image first."}), 400
    if not zone:
        return jsonify({"error": "Enter a zone name (for example A or B)."}), 400

    tmp = {"image": TMP / f"{token}.jpg", "overlay": TMP / f"{token}_overlay.jpg",
           "mask": TMP / f"{token}_mask.png"}
    if not all(p.exists() for p in tmp.values()):
        return jsonify({"error": "This analysis is no longer available. Capture it again."}), 400

    mask = cv2.imread(str(tmp["mask"]), cv2.IMREAD_GRAYSCALE)
    ratio = float((mask > 0).mean())    # recomputed here, never trusted from the browser
    now = datetime.now()
    row_id = run("""INSERT INTO inspections
        (date, time, zone, notes, corrosion_ratio, model, image_file, overlay_file, mask_file)
        VALUES (?,?,?,?,?,?,'','','')""",
                 (now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), zone, notes, ratio, model_name()))
    code = f"INSP-{row_id:04d}"
    names = {"image_file": f"images/{code}.jpg",
             "overlay_file": f"results/{code}_overlay.jpg",
             "mask_file": f"results/{code}_mask.png"}
    try:
        shutil.move(str(tmp["image"]), DATA / names["image_file"])
        shutil.move(str(tmp["overlay"]), DATA / names["overlay_file"])
        shutil.move(str(tmp["mask"]), DATA / names["mask_file"])
    except Exception as e:
        run("DELETE FROM inspections WHERE id=?", (row_id,))
        return jsonify({"error": f"Could not save the files: {e}"}), 500
    run("UPDATE inspections SET code=?, image_file=?, overlay_file=?, mask_file=? WHERE id=?",
        (code, names["image_file"], names["overlay_file"], names["mask_file"], row_id))
    return jsonify(present(fetch("SELECT * FROM inspections WHERE id=?", (row_id,))[0]))


@app.route("/api/inspections/<int:row_id>/review", methods=["POST"])
def api_review(row_id):
    body = request.get_json(silent=True) or {}
    status = body.get("status")
    if status not in REVIEW_STATES:
        return jsonify({"error": "Status must be pending, accepted or rejected."}), 400
    if not fetch("SELECT id FROM inspections WHERE id=?", (row_id,)):
        return jsonify({"error": "Inspection not found."}), 404
    comment = str(body.get("comment", "")).strip()[:1000]
    run("UPDATE inspections SET review_status=?, review_comment=?, reviewed_at=? WHERE id=?",
        (status, comment, datetime.now().strftime("%Y-%m-%d %H:%M"), row_id))
    return jsonify(present(fetch("SELECT * FROM inspections WHERE id=?", (row_id,))[0]))


init_db()
clean_tmp()

if __name__ == "__main__":
    threading.Thread(target=load_model_in_background, daemon=True).start()
    print("Open http://127.0.0.1:5000 in your browser (Ctrl+C to stop)")
    # host 127.0.0.1 = only this computer. There is no login, so do not open it to a network.
    app.run(host="127.0.0.1", port=5000, debug=False)