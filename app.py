"""
Maize Disease API — Render deployment
======================================
Image inference is forwarded to the Hugging Face Space:
    https://dracer-crop-desease-detection.hf.space/predict

No model files needed on Render. The response is mapped back to the
existing format so the ESP32 firmware and dashboard need zero changes.

Everything else is identical to the original app:
  - /predict     → receives image + sensor data from ESP32
  - /latest      → last reading
  - /history     → recent readings
  - /dashboard   → serves static/dashboard.html
  - /            → redirects to /dashboard
"""

import io, base64, os, logging
from datetime import datetime, timezone
from collections import deque

import requests                         # HTTP forwarding to HF Space
from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS

# ── Config ────────────────────────────────────────────────────────────────────
# Set HF_SPACE_URL as an environment variable on Render if you ever want to
# switch endpoints without redeploying.
HF_SPACE_URL = os.environ.get(
    "HF_SPACE_URL",
    "https://dracer-crop-desease-detection.hf.space/predict"
)
HF_TIMEOUT = 30   # seconds — HF free tier can cold-start in ~15 s

# Pest thresholds (unchanged)
PEST_TEMP_MIN     = 24.0
PEST_TEMP_MAX     = 32.0
PEST_HUMIDITY_MIN = 60.0
GAS_THRESHOLD     = 1027

HISTORY_MAXLEN = 50

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("maize-proxy")

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

history        = deque(maxlen=HISTORY_MAXLEN)
latest_reading = None

# ── Helpers ───────────────────────────────────────────────────────────────────
def parse_float(v):
    try:    return float(v)
    except: return None

def assess_pest_risk(temperature, humidity, gas):
    reasons = []
    temp_ok = temperature is not None and PEST_TEMP_MIN <= temperature <= PEST_TEMP_MAX
    hum_ok  = humidity    is not None and humidity >= PEST_HUMIDITY_MIN
    gas_ok  = gas         is not None and gas      >= GAS_THRESHOLD
    if temp_ok: reasons.append(f"Temperature {temperature:.1f}°C is in the pest-favorable range")
    if hum_ok:  reasons.append(f"Humidity {humidity:.1f}% is above the {PEST_HUMIDITY_MIN:.0f}% threshold")
    if gas_ok:  reasons.append(f"Gas reading {gas:.0f} is above the {GAS_THRESHOLD:.0f} threshold")
    return ("HIGH" if (temp_ok and (hum_ok or gas_ok)) else "LOW"), reasons

def map_hf_response(hf_data):
    """
    Maps whatever the HF Space returns into the existing response keys
    the ESP32 and dashboard already expect.

    HF Space may return (cassava-style):
        label, confidence, status, is_leaf, advice, all_probs ...
    Or a simpler format:
        disease, confidence, status ...

    We handle both gracefully.
    """
    # Disease label — try both key names
    disease = (hf_data.get("label")
               or hf_data.get("disease")
               or "Unknown")

    # Confidence — strip "%" if it's already a string, else use float
    raw_conf = hf_data.get("confidence", 0)
    if isinstance(raw_conf, str):
        conf_str = raw_conf if raw_conf.endswith("%") else raw_conf + "%"
        conf_float = float(raw_conf.strip("%"))
    else:
        conf_float = float(raw_conf)
        conf_str   = f"{conf_float:.1f}%"

    # Status → map to ALERT / OK / NO_CROP
    hf_status = hf_data.get("status", "")
    if hf_status in ("healthy", "OK"):
        status = "OK"
    elif hf_status == "no_leaf":
        status = "NO_CROP"
    else:
        # sick / ALERT / anything else
        status = "ALERT" if disease.lower() not in ("healthy", "unknown") else "OK"

    return disease, conf_str, status

# ── /predict ──────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    global latest_reading

    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    img_bytes   = request.files["image"].read()
    temperature = parse_float(request.form.get("temperature"))
    humidity    = parse_float(request.form.get("humidity"))
    gas         = parse_float(request.form.get("gas_raw"))

    pest_risk, pest_reasons = assess_pest_risk(temperature, humidity, gas)
    img_b64 = "data:image/jpeg;base64," + base64.b64encode(img_bytes).decode()
    ts      = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # ── Forward image to HF Space ─────────────────────────────────────────────
    try:
        hf_resp = requests.post(
            HF_SPACE_URL,
            files={"image": ("capture.jpg", img_bytes, "image/jpeg")},
            timeout=HF_TIMEOUT,
        )
        hf_resp.raise_for_status()
        hf_data = hf_resp.json()
        log.info(f"HF Space responded: {hf_data.get('label') or hf_data.get('disease')} "
                 f"({hf_data.get('confidence')})")
    except requests.exceptions.Timeout:
        log.error("HF Space timed out")
        return jsonify({"error": "Inference service timed out. Try again in a moment."}), 504
    except requests.exceptions.RequestException as e:
        log.error(f"HF Space request failed: {e}")
        return jsonify({"error": f"Inference service unavailable: {str(e)}"}), 502
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        return jsonify({"error": str(e)}), 500

    # ── Map HF response to existing format ────────────────────────────────────
    disease, conf_str, status = map_hf_response(hf_data)

    result = {
        # ── Exact same keys the ESP32 + dashboard already read ────────────────
        "disease"      : disease,
        "confidence"   : conf_str,
        "status"       : status,
        "temperature"  : temperature,
        "humidity"     : humidity,
        "gas"          : gas,
        "pest_risk"    : pest_risk,
        "pest_reasons" : pest_reasons,
        "timestamp"    : ts,
        "image"        : img_b64,
        # ── Extra fields passed through from HF (additive — old clients ignore) ─
        "advice"       : hf_data.get("advice", ""),
        "is_leaf"      : hf_data.get("is_leaf", True),
        "all_probs"    : hf_data.get("all_probs", {}),
    }

    latest_reading = result
    history.appendleft(result)

    log.info(f"Prediction: {disease} ({conf_str}) | "
             f"T={temperature} H={humidity} Gas={gas} | Pest risk: {pest_risk}")

    return jsonify(result)

# ── Original polling routes (all unchanged) ───────────────────────────────────
@app.route("/latest", methods=["GET"])
def get_latest():
    if latest_reading is None:
        return jsonify({"error": "No readings yet"}), 404
    return jsonify(latest_reading)

@app.route("/history", methods=["GET"])
def get_history():
    return jsonify(list(history))

@app.route("/dashboard", methods=["GET"])
def dashboard():
    return send_from_directory("static", "dashboard.html")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "inference": "hf_space", "url": HF_SPACE_URL})

@app.route("/", methods=["GET"])
def home():
    return redirect("/dashboard")

# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
