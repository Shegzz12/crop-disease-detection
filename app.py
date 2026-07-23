# ── backend/app.py ────────────────────────────────────────────────────────────
"""
FarmBot Backend — ONNX Runtime edition (models cached from Hugging Face)
Runs two models in sequence:
  1. Gate model     (gate_model.onnx) → is there a maize leaf in the image?
  2. Disease model (best.onnx)        → 102-way IP102 pest classifier.
"""
import os, io, time, logging, sqlite3, uuid, json
import urllib.request
from datetime import datetime

import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import onnxruntime as ort

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GATE_MODEL_URL    = "https://huggingface.co/Samson123Ade/maize-infection-detection/resolve/main/gate_model.onnx"
DISEASE_MODEL_URL = "https://huggingface.co/Samson123Ade/maize-infection-detection/resolve/main/best.onnx"

GATE_LOCAL_PATH    = os.path.join(BASE_DIR, "models", "gate_model.onnx")
DISEASE_LOCAL_PATH = os.path.join(BASE_DIR, "models", "best.onnx")

CATEGORY_FILE = os.path.join(BASE_DIR, "categories.json")
CATEGORY_MAP  = {}

FALLBACK_LABEL = "Unclassified Detection"
FALLBACK_ADVICE = {
    "cultural_biological": (
        "The affected area doesn't match a specific pest in our maize catalog. "
        "Isolate/inspect the plant, remove visibly damaged foliage, and monitor "
        "for spread before treating."
    ),
    "chemical_direct": (
        "Hold off on a specific chemical until the exact cause is confirmed — "
        "consult a local agronomist or extension officer with a close-up photo."
    ),
}

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH    = os.path.join(BASE_DIR, "scans.db")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# ── Model Resolution & Threshold Settings ──────────────────────────────────────
GATE_SIZE       = 224
DISEASE_SIZE    = 224
GATE_THRESH     = 0.30     # confidence threshold for leaf detection

# ⚠️ CLASS INDEX ADJUSTMENT:
# Set GATE_LEAF_INDEX = 0 if index 0 represents "leaf".
# Set GATE_LEAF_INDEX = 1 if index 1 represents "leaf".
GATE_LEAF_INDEX = 0

# Set COLOR_MODE to "BGR" if the gate model was trained using OpenCV/Albumentations defaults
COLOR_MODE      = "BGR"    # "RGB" or "BGR"

MIN_CONFIDENCE  = 40.0

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PORT = int(os.environ.get("PORT", 5500))
HOST = "0.0.0.0"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("farmbot")

model_cache_state = {
    "gate_cached_at": None,
    "disease_cached_at": None,
}

# ── Categories ────────────────────────────────────────────────────────────────
def load_categories():
    global CATEGORY_MAP
    if os.path.exists(CATEGORY_FILE):
        try:
            with open(CATEGORY_FILE, "r", encoding="utf-8") as f:
                CATEGORY_MAP = json.load(f)
            log.info(f"Loaded {len(CATEGORY_MAP)} categories from categories.json")
        except Exception as e:
            log.warning(f"Failed to load categories.json: {e}")

load_categories()

# ── Download + Load ONNX Models ───────────────────────────────────────────────
def load_onnx_model(url, local_path, name):
    if not os.path.exists(local_path):
        log.info(f"Downloading {name} model from Hugging Face...")
        urllib.request.urlretrieve(url, local_path)
    
    cached_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if name == "gate":
        model_cache_state["gate_cached_at"] = cached_at
    elif name == "disease":
        model_cache_state["disease_cached_at"] = cached_at
    
    session = ort.InferenceSession(local_path, providers=["CPUExecutionProvider"])
    
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    log.info(f"[{name.upper()}] Expected Input: {input_shape} | Output: {output_shape}")
    
    return session

log.info("Loading gate model ...")
gate_session = load_onnx_model(GATE_MODEL_URL, GATE_LOCAL_PATH, "gate")
log.info("Loading disease model ...")
disease_session = load_onnx_model(DISEASE_MODEL_URL, DISEASE_LOCAL_PATH, "disease")

# ── Flask + Database Init ─────────────────────────────────────────────────────
app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            filename          TEXT NOT NULL,
            is_leaf           INTEGER,
            leaf_conf         REAL,
            label             TEXT,
            class_id          INTEGER,
            identified        INTEGER,
            status            TEXT,
            severity          TEXT,
            color             TEXT,
            confidence        REAL,
            temperature       REAL,
            humidity          REAL,
            gas_raw           TEXT,
            gas_voltage       TEXT,
            cultural_biological TEXT,
            chemical_direct     TEXT,
            device_time       TEXT,
            server_time       TEXT,
            inference_ms      REAL,
            all_probs         TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ── Preprocessing ─────────────────────────────────────────────────────────────
def bytes_to_input(file_bytes, size, color_mode="RGB"):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize((size, size))
    arr = np.array(img).astype(np.float32)
    
    if color_mode == "BGR":
        arr = arr[:, :, ::-1]  # RGB to BGR
        
    arr = arr / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = np.transpose(arr, (2, 0, 1))          # HWC -> CHW
    arr = np.expand_dims(arr, axis=0).astype(np.float32)
    return arr

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def run_session(session, input_tensor):
    input_name  = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    raw = session.run([output_name], {input_name: input_tensor})[0][0]
    return softmax(raw)

def resolve_prediction(probabilities):
    pred_id = int(np.argmax(probabilities))
    confidence = round(float(probabilities[pred_id]) * 100, 2)
    entry = CATEGORY_MAP.get(str(pred_id))

    if entry and confidence >= MIN_CONFIDENCE:
        return {
            "identified": True,
            "class_id": pred_id,
            "label": entry["problem"],
            "confidence": confidence,
            "cultural_biological": entry["cultural_biological"],
            "chemical_direct": entry["chemical_direct"],
        }
    return {
        "identified": False,
        "class_id": pred_id,
        "label": FALLBACK_LABEL,
        "confidence": confidence,
        "cultural_biological": FALLBACK_ADVICE["cultural_biological"],
        "chemical_direct": FALLBACK_ADVICE["chemical_direct"],
    }

def classify_severity(confidence, identified):
    if not identified:
        return {"status": "unclassified", "severity": "unknown", "color": "orange"}
    severity = "mild" if confidence < 60 else "severe"
    color    = "orange" if severity == "mild" else "red"
    return {"status": "sick", "severity": severity, "color": color}

def calculate_pest_risk(confidence, identified, temperature, humidity, gas_raw):
    pest_reasons = []
    if temperature is not None:
        try:
            t = float(temperature)
            if 24 <= t <= 32:
                pest_reasons.append(f"Temperature {t:.1f}°C favors pest activity")
        except (ValueError, TypeError):
            pass
    
    if humidity is not None:
        try:
            h = float(humidity)
            if h >= 60:
                pest_reasons.append(f"Humidity {h:.1f}% supports disease development")
        except (ValueError, TypeError):
            pass
            
    risk = "HIGH" if identified and len(pest_reasons) >= 2 else "LOW"
    return risk, pest_reasons

def transform_response_to_frontend(response, temperature=None, humidity=None, gas_raw=None, gas_voltage=None):
    transformed = {
        "disease": response.get("label", "Unknown"),
        "label": response.get("label", "Unknown"),
        "class_id": response.get("class_id"),
        "identified": response.get("identified", False),
        "confidence": response.get("confidence", 0),
        "status": response.get("status", "unknown"),
        "severity": response.get("severity", "unknown"),
        "color": response.get("color", "gray"),
        "is_leaf": response.get("is_leaf", False),
        "leaf_confidence": response.get("leaf_confidence", 0),
        "pest_risk": "LOW",
        "pest_reasons": [],
        "cultural_biological": response.get("cultural_biological", ""),
        "chemical_direct": response.get("chemical_direct", ""),
        "temperature": temperature,
        "humidity": humidity,
        "gas": gas_raw,
        "gas_voltage": gas_voltage,
        "timestamp": response.get("server_timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "inference_ms": response.get("inference_ms", 0),
        "all_probs": response.get("all_probs", {}),
    }
    
    try:
        risk, reasons = calculate_pest_risk(
            response.get("confidence", 0),
            response.get("identified", False),
            temperature,
            humidity,
            gas_raw
        )
        transformed["pest_risk"] = risk
        transformed["pest_reasons"] = reasons
    except Exception as e:
        log.warning(f"Error calculating pest risk: {e}")
    
    return transformed

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": gate_session is not None and disease_session is not None,
        "categories_count": len(CATEGORY_MAP),
    })

@app.route("/predict", methods=["POST"])
def predict():
    start = time.time()

    if "image" not in request.files:
        return jsonify({"error": "No image field in request."}), 400
    file_bytes  = request.files["image"].read()
    device_time = request.form.get("captured_at", "").strip()
    if not file_bytes:
        return jsonify({"error": "Empty image."}), 400
    
    temperature = request.form.get("temperature")
    humidity    = request.form.get("humidity")
    gas_raw     = request.form.get("gas_raw")
    gas_voltage = request.form.get("gas_voltage")
    
    try:
        # ── Stage 1: Gate Model Check ──────────────────────────────────────────
        g_input   = bytes_to_input(file_bytes, GATE_SIZE, color_mode=COLOR_MODE)
        g_probs   = run_session(gate_session, g_input)
        
        # Log both class outputs to clarify index mapping
        log.info(f"Gate Output Raw Probs -> Class 0: {g_probs[0]:.4f} | Class 1: {g_probs[1]:.4f}")
        
        leaf_conf = float(g_probs[GATE_LEAF_INDEX])
        is_leaf   = leaf_conf >= GATE_THRESH

        if not is_leaf:
            elapsed = round((time.time() - start) * 1000, 1)
            log.info(f"Gate rejected (leaf_conf={leaf_conf:.2f}) in {elapsed}ms")
            raw_response = {
                "is_leaf":         False,
                "leaf_confidence": round(leaf_conf * 100, 1),
                "label":           "No Leaf Detected",
                "class_id":        None,
                "identified":      False,
                "status":          "no_leaf",
                "severity":        "none",
                "color":           "gray",
                "confidence":      round((1 - leaf_conf) * 100, 1),
                "cultural_biological": "No maize leaf detected. Point the camera directly at the leaf.",
                "chemical_direct":     "—",
                "server_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "inference_ms":      elapsed,
            }
            response = transform_response_to_frontend(raw_response, temperature, humidity, gas_raw, gas_voltage)
            _save_scan(file_bytes, raw_response, device_time, temperature, humidity, gas_raw, gas_voltage)
            return jsonify(response), 200

        # ── Stage 2: Disease Classification ───────────────────────────────────
        d_input  = bytes_to_input(file_bytes, DISEASE_SIZE, color_mode="RGB")
        d_probs  = run_session(disease_session, d_input)
        result   = resolve_prediction(d_probs)

        all_probs = {}
        for k, v in CATEGORY_MAP.items():
            try:
                idx = int(k)
                if idx < len(d_probs):
                    all_probs[v["problem"]] = round(float(d_probs[idx]) * 100, 2)
            except (ValueError, IndexError):
                continue

        diag    = classify_severity(result["confidence"], result["identified"])
        elapsed = round((time.time() - start) * 1000, 1)

        raw_response = {
            "is_leaf":          True,
            "leaf_confidence":  round(leaf_conf * 100, 1),
            "label":            result["label"],
            "class_id":         result["class_id"],
            "identified":       result["identified"],
            "status":           diag["status"],
            "severity":         diag["severity"],
            "color":            diag["color"],
            "confidence":       result["confidence"],
            "cultural_biological": result["cultural_biological"],
            "chemical_direct":     result["chemical_direct"],
            "all_probs":        all_probs,
            "server_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "inference_ms":      elapsed,
        }
        log.info(f"Predicted: {result['label']} ({result['confidence']}%, identified={result['identified']}) in {elapsed}ms")
        
        response = transform_response_to_frontend(raw_response, temperature, humidity, gas_raw, gas_voltage)
        _save_scan(file_bytes, raw_response, device_time, temperature, humidity, gas_raw, gas_voltage)
        return jsonify(response), 200

    except Exception as e:
        log.exception("Prediction error")
        return jsonify({"error": str(e)}), 500

def _save_scan(file_bytes, resp, device_time, temperature=None, humidity=None, gas_raw=None, gas_voltage=None):
    try:
        filename = f"scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
        with open(os.path.join(UPLOAD_DIR, filename), "wb") as f:
            f.write(file_bytes)
        conn = get_db()
        conn.execute("""
            INSERT INTO scans
            (filename, is_leaf, leaf_conf, label, class_id, identified, status,
             severity, color, confidence, temperature, humidity, gas_raw, gas_voltage,
             cultural_biological, chemical_direct,
             device_time, server_time, inference_ms, all_probs)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            filename,
            1 if resp.get("is_leaf") else 0,
            resp.get("leaf_confidence"),
            resp.get("label"),
            resp.get("class_id"),
            1 if resp.get("identified") else 0,
            resp.get("status"),
            resp.get("severity"),
            resp.get("color"),
            resp.get("confidence"),
            temperature,
            humidity,
            gas_raw,
            gas_voltage,
            resp.get("cultural_biological"),
            resp.get("chemical_direct"),
            device_time,
            resp.get("server_timestamp"),
            resp.get("inference_ms"),
            json.dumps(resp.get("all_probs", {})),
        ))
        conn.commit()
        conn.close()
    except Exception:
        log.exception("Failed to save scan")

@app.route("/latest")
def api_latest():
    conn = get_db()
    row = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    conn.close()
    
    if not row:
        return jsonify({"error": "no scans yet"}), 404
    
    item = dict(row)
    item["image_url"] = f"/uploads/{row['filename']}"
    try:
        item["all_probs"] = json.loads(row["all_probs"] or "{}")
    except Exception:
        item["all_probs"] = {}
    
    transformed = {
        "disease": item["label"],
        "label": item["label"],
        "class_id": item["class_id"],
        "identified": bool(item["identified"]),
        "confidence": item["confidence"],
        "status": item["status"],
        "severity": item["severity"],
        "color": item["color"],
        "is_leaf": bool(item["is_leaf"]),
        "leaf_confidence": item["leaf_conf"],
        "temperature": item["temperature"],
        "humidity": item["humidity"],
        "gas": item["gas_raw"],
        "gas_voltage": item["gas_voltage"],
        "timestamp": item["server_time"],
        "inference_ms": item["inference_ms"],
        "all_probs": item["all_probs"],
        "cultural_biological": item["cultural_biological"],
        "chemical_direct": item["chemical_direct"],
        "image_url": item["image_url"],
    }
    
    risk, reasons = calculate_pest_risk(
        item["confidence"],
        bool(item["identified"]),
        item["temperature"],
        item["humidity"],
        item["gas_raw"]
    )
    transformed["pest_risk"] = risk
    transformed["pest_reasons"] = reasons
    
    return jsonify(transformed), 200

@app.route("/api/history")
def api_history():
    limit  = min(int(request.args.get("limit", 24)), 200)
    offset = int(request.args.get("offset", 0))
    status = request.args.get("status", "all")

    conn = get_db()
    if status in ("sick", "unclassified", "no_leaf"):
        rows  = conn.execute("SELECT * FROM scans WHERE status=? ORDER BY id DESC LIMIT ? OFFSET ?", (status, limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM scans WHERE status=?", (status,)).fetchone()[0]
    else:
        rows  = conn.execute("SELECT * FROM scans ORDER BY id DESC LIMIT ? OFFSET ?", (limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    conn.close()

    items = []
    for r in rows:
        item = dict(r)
        item["image_url"] = f"/uploads/{r['filename']}"
        try:
            item["all_probs"] = json.loads(r["all_probs"] or "{}")
        except Exception:
            item["all_probs"] = {}
        
        transformed = {
            "disease": item["label"],
            "label": item["label"],
            "class_id": item["class_id"],
            "identified": bool(item["identified"]),
            "confidence": item["confidence"],
            "status": item["status"],
            "severity": item["severity"],
            "color": item["color"],
            "is_leaf": bool(item["is_leaf"]),
            "leaf_confidence": item["leaf_conf"],
            "temperature": item["temperature"],
            "humidity": item["humidity"],
            "gas": item["gas_raw"],
            "gas_voltage": item["gas_voltage"],
            "timestamp": item["server_time"],
            "inference_ms": item["inference_ms"],
            "all_probs": item["all_probs"],
            "cultural_biological": item["cultural_biological"],
            "chemical_direct": item["chemical_direct"],
            "image_url": item["image_url"],
        }
        
        risk, reasons = calculate_pest_risk(
            item["confidence"],
            bool(item["identified"]),
            item["temperature"],
            item["humidity"],
            item["gas_raw"]
        )
        transformed["pest_risk"] = risk
        transformed["pest_reasons"] = reasons
        items.append(transformed)

    return jsonify({"items": items, "total": total, "limit": limit, "offset": offset})

@app.route("/api/stats")
def api_stats():
    conn = get_db()
    total        = conn.execute("SELECT COUNT(*) FROM scans").fetchone()[0]
    sick         = conn.execute("SELECT COUNT(*) FROM scans WHERE status='sick'").fetchone()[0]
    unclassified = conn.execute("SELECT COUNT(*) FROM scans WHERE status='unclassified'").fetchone()[0]
    no_leaf      = conn.execute("SELECT COUNT(*) FROM scans WHERE status='no_leaf'").fetchone()[0]
    last         = conn.execute("SELECT COALESCE(device_time, server_time) FROM scans ORDER BY id DESC LIMIT 1").fetchone()
    avg_c        = conn.execute("SELECT AVG(confidence) FROM scans WHERE is_leaf=1").fetchone()[0]
    conn.close()
    return jsonify({
        "total":            total,
        "sick":             sick,
        "unclassified":     unclassified,
        "no_leaf":          no_leaf,
        "sick_pct":         round(sick / total * 100, 1) if total else 0,
        "unclassified_pct": round(unclassified / total * 100, 1) if total else 0,
        "avg_confidence":   round(avg_c, 1) if avg_c else 0,
        "last_scan_time":   last[0] if last else None,
    })

@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.route("/")
def serve_dashboard():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=True)
