import asyncio
import json
import logging
import threading
import uuid
import concurrent.futures
from typing import Dict, Any
import os
import io
import time
import base64
import uuid
from PIL import Image  # pip install pillow
from flask import send_from_directory

import websockets
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("sim-server")

# ---------------------------
# Flask app + CORS
# ---------------------------
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------------------
# Globals / state
# ---------------------------
connected = set()          # set of active websocket connections
async_loop = None          # will be set to the asyncio loop running websockets server
collision_count = 0        # server-tracked collisions
latest_capture = None      # fallback if a capture response arrives without request_id
capture_waits: Dict[str, asyncio.Future] = {}  # maps request_id -> Future

# Canvas / coordinate helpers
FLOOR_HALF = 50

# Directory to save captures
CAPTURE_DIR = os.path.join(os.path.dirname(__file__), "captures")
os.makedirs(CAPTURE_DIR, exist_ok=True)


def _safe_filename(prefix="capture", ext="png"):
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}.{ext}"

@app.route('/upload_capture', methods=['POST', 'OPTIONS'])
def upload_capture():
    """
    Accepts multipart/form-data 'image' (Blob/png). Also accepts JSON {image: 'data:image/png;base64,...'} as fallback.
    Saves PNG and generates a PDF conversion, returns JSON with filenames and download URLs.
    """
    if request.method == 'OPTIONS':
        return _ok_options_response()

    # Try multipart/form-data file first
    file = None
    if 'image' in request.files:
        file = request.files['image']
        try:
            data = file.read()
        except Exception as e:
            return jsonify({'error': f'Failed to read uploaded file: {e}'}), 400
        png_bytes = data
    else:
        # fallback: JSON base64 data-url (not recommended for large images)
        data_json = request.get_json(silent=True) or {}
        img_data = data_json.get('image') or data_json.get('data')  # support different keys
        if not img_data:
            return jsonify({'error': "No image found in uploaded files or JSON body"}), 400
        if isinstance(img_data, str) and img_data.startswith("data:image"):
            header, b64 = img_data.split(',', 1)
            try:
                png_bytes = base64.b64decode(b64)
            except Exception as e:
                return jsonify({'error': f'Bad base64 image: {e}'}), 400
        else:
            return jsonify({'error': 'Unsupported image payload format'}), 400

    # Save PNG
    png_name = _safe_filename("capture", "png")
    png_path = os.path.join(CAPTURE_DIR, png_name)
    try:
        with open(png_path, "wb") as f:
            f.write(png_bytes)
    except Exception as e:
        return jsonify({'error': f'Failed to save PNG: {e}'}), 500

    # Convert PNG to PDF using Pillow
    pdf_name = png_name.rsplit('.', 1)[0] + ".pdf"
    pdf_path = os.path.join(CAPTURE_DIR, pdf_name)
    try:
        image = Image.open(io.BytesIO(png_bytes))
        # Convert RGBA to RGB if needed (PDF needs RGB)
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and 'transparency' in image.info):
            # Create a white background
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3] if image.mode == 'RGBA' else None)
            image = bg
        else:
            image = image.convert("RGB")
        image.save(pdf_path, "PDF", resolution=100.0)
    except Exception as e:
        # If PDF conversion fails, still return PNG info
        log.exception("PDF conversion failed")
        return jsonify({
            'status': 'saved_png_only',
            'png_filename': png_name,
            'png_path': png_path,
            'error': str(e)
        }), 500

    # Return JSON with filenames and accessible URLs
    download_url_png = f"/captures/{png_name}"
    download_url_pdf = f"/captures/{pdf_name}"
    return jsonify({
        'status': 'saved',
        'png_filename': png_name,
        'pdf_filename': pdf_name,
        'png_url': download_url_png,
        'pdf_url': download_url_pdf
    })

@app.route('/last_capture', methods=['GET'])
def last_capture():
    global latest_capture
    if latest_capture is None:
        return jsonify({'error': 'no capture available'}), 404
    # Return the stored dict as-is (may contain 'image' data-url)
    return jsonify(latest_capture)


def corner_to_coords(corner: str, margin=5):
    c = corner.upper()
    x = FLOOR_HALF - margin if "E" in c else -(FLOOR_HALF - margin)
    z = FLOOR_HALF - margin if ("S" in c or "B" in c) else -(FLOOR_HALF - margin)
    if c in ("NE", "EN", "TR"): x, z = (FLOOR_HALF - margin, -(FLOOR_HALF - margin))
    if c in ("NW", "WN", "TL"): x, z = (-(FLOOR_HALF - margin), -(FLOOR_HALF - margin))
    if c in ("SE", "ES", "BR"): x, z = (FLOOR_HALF - margin, (FLOOR_HALF - margin))
    if c in ("SW", "WS", "BL"): x, z = (-(FLOOR_HALF - margin), (FLOOR_HALF - margin))
    return {"x": float(x), "y": 0.0, "z": float(z)}

# ---------------------------
# WebSocket handler and helpers
# ---------------------------
async def ws_handler(websocket, path=None):
    """Handle messages coming from simulator (browser client)."""
    global collision_count, latest_capture
    log.info("WebSocket client connected")
    connected.add(websocket)
    try:
        async for message in websocket:
            log.debug(f"Received WS message: {message[:200]}")
            try:
                data = json.loads(message)
                log.info(f"ws_handler: received message keys={list(data.keys())}")
            except json.JSONDecodeError:
                log.warning("Non-JSON message received via WebSocket")
                continue

            # collision messages
            if isinstance(data, dict) and data.get("type") == "collision" and data.get("collision"):
                collision_count += 1
                log.info(f"Collision reported by simulator. total collisions={collision_count}")

            # capture responses: prefer to resolve a waiting future by request_id
            if isinstance(data, dict) and data.get("type") == "capture_image_response":
                log.info(f"ws_handler: got capture_image_response request_id={data.get('request_id')} image_present={bool(data.get('image'))}")
                # Resolve a per-request future if present
                rid = data.get("request_id")
                if rid:
                    fut = capture_waits.pop(rid, None)
                    if fut and not fut.done():
                        fut.set_result(data)
                        # also store it for /last_capture
                        latest_capture = data
                        log.info("ws_handler: resolved future and set latest_capture (by request_id)")
                        continue
                # fallback: store as latest_capture
                latest_capture = data
                log.info("ws_handler: stored latest_capture (no request_id match)")

    except websockets.exceptions.ConnectionClosed:
        log.info("WebSocket client disconnected")
    finally:
        connected.discard(websocket)


def broadcast(msg: Dict[str, Any]) -> bool:
    """Broadcast a JSON-serializable message to all connected websockets.
    Returns False if no clients connected or loop missing.
    """
    global async_loop
    if not connected:
        log.debug("Broadcast: no connected simulators")
        return False
    if async_loop is None:
        log.warning("Broadcast: asyncio loop not ready")
        return False

    payload = json.dumps(msg)
    for ws in list(connected):
        try:
            asyncio.run_coroutine_threadsafe(ws.send(payload), async_loop)
        except Exception as e:
            log.exception("Failed to schedule WS send: %s", e)
    log.debug("Broadcasted message to %d clients: %s", len(connected), msg.get("command"))
    return True

# ---------------------------
# Helper to return JSON for OPTIONS preflight
# ---------------------------

def _ok_options_response():
    resp = make_response(jsonify({"status": "ok"}), 200)
    return resp

# ---------------------------
# Flask REST endpoints
# ---------------------------
@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "ok",
        "ws_clients": len(connected),
        "collisions": collision_count
    })

@app.route('/move', methods=['POST', 'OPTIONS'])
def move():
    if request.method == 'OPTIONS':
        return _ok_options_response()
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({'error': 'Missing JSON body'}), 400
    if 'x' not in data or 'z' not in data:
        return jsonify({'error': 'Missing parameters. Please provide "x" and "z".'}), 400
    x, z = float(data['x']), float(data['z'])
    msg = {"command": "move", "target": {"x": x, "y": float(data.get('y', 0.0)), "z": z}}
    ok = broadcast(msg)
    if not ok:
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'move command sent', 'command': msg})

@app.route('/move_rel', methods=['POST', 'OPTIONS'])
def move_rel():
    if request.method == 'OPTIONS':
        return _ok_options_response()
    data = request.get_json(force=True, silent=True)
    if not data or 'turn' not in data or 'distance' not in data:
        return jsonify({'error': 'Missing parameters. Please provide "turn" and "distance".'}), 400
    msg = {"command": "move_relative", "turn": float(data['turn']), "distance": float(data['distance'])}
    ok = broadcast(msg)
    if not ok:
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'move relative command sent', 'command': msg})

@app.route('/stop', methods=['POST', 'OPTIONS'])
def stop():
    if request.method == 'OPTIONS':
        return _ok_options_response()
    msg = {"command": "stop"}
    ok = broadcast(msg)
    if not ok:
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'stop command sent', 'command': msg})

# ---------------------------
# CAPTURE: use async_loop and a per-request future
# ---------------------------
@app.route('/capture', methods=['GET', 'OPTIONS'])
def capture():
    if request.method == 'OPTIONS':
        return _ok_options_response()

    global async_loop, capture_waits, latest_capture
    if async_loop is None:
        return jsonify({'error': 'WebSocket server not running'}), 503

    # clear any previous capture so fallback won't return stale data
    latest_capture = None
    log.info("capture: cleared latest_capture before sending new request")

    # create a unique request id and future on the async loop
    request_id = str(uuid.uuid4())
    fut = async_loop.create_future()
    capture_waits[request_id] = fut

    # broadcast capture request including request_id
    msg = {"command": "capture_image", "request_id": request_id}
    ok = broadcast(msg)
    if not ok:
        capture_waits.pop(request_id, None)
        return jsonify({'error': 'No connected simulators.'}), 400

    # Wait for the future to be set by ws_handler (timeout protects the HTTP thread)
    try:
        wait_coro = asyncio.wait_for(fut, timeout=8.0)   # increase timeout if needed
        result = asyncio.run_coroutine_threadsafe(wait_coro, async_loop).result(timeout=9.0)

        # store the fresh result
        latest_capture = result
        log.info(f"Capture completed for request_id={request_id}, image_present={bool(result.get('image'))}")

        # Return concise metadata
        return jsonify({
            'status': 'capture returned',
            'timestamp': result.get('timestamp'),
            'position': result.get('position'),
            'image_present': bool(result.get('image'))
        })
    except concurrent.futures.TimeoutError:
        capture_waits.pop(request_id, None)
        # fallback: if latest_capture exists, return it (best-effort)
        if latest_capture is not None:
            return jsonify({
                'status': 'timeout_but_have_latest',
                'timestamp': latest_capture.get('timestamp'),
                'image_present': bool(latest_capture.get('image'))
            })
        return jsonify({'error': 'Timeout waiting for capture'}), 504
    except Exception as e:
        capture_waits.pop(request_id, None)
        return jsonify({'error': str(e)}), 500

@app.route('/goal', methods=['POST', 'OPTIONS'])
def set_goal():
    if request.method == 'OPTIONS':
        return _ok_options_response()
    data = request.get_json(force=True, silent=True) or {}
    if 'corner' in data:
        pos = corner_to_coords(str(data['corner']))
    elif 'x' in data and 'z' in data:
        pos = {"x": float(data['x']), "y": float(data.get('y', 0.0)), "z": float(data['z'])}
    else:
        return jsonify({'error': 'Provide {"corner":"NE|NW|SE|SW"} OR {"x":..,"z":..}'}), 400
    msg = {"command": "set_goal", "position": pos}
    ok = broadcast(msg)
    if not ok:
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'goal set', 'goal': pos})

@app.route('/obstacles/positions', methods=['POST', 'OPTIONS'])
def set_obstacle_positions():
    if request.method == 'OPTIONS':
        return _ok_options_response()
    data = request.get_json(force=True, silent=True) or {}
    positions = data.get('positions')
    if not isinstance(positions, list) or not positions:
        return jsonify({'error': 'Provide "positions" as a non-empty list.'}), 400
    norm = []
    for p in positions:
        if not isinstance(p, dict) or 'x' not in p or 'z' not in p:
            return jsonify({'error': 'Each position needs "x" and "z".'}), 400
        norm.append({"x": float(p['x']), "y": float(p.get('y', 2.0)), "z": float(p['z'])})
    msg = {"command": "set_obstacles", "positions": norm}
    ok = broadcast(msg)
    if not ok:
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'obstacles updated', 'count': len(norm)})

@app.route('/obstacles/motion', methods=['POST', 'OPTIONS'])
def set_obstacle_motion():
    if request.method == 'OPTIONS':
        return _ok_options_response()
    data = request.get_json(force=True, silent=True) or {}
    if 'enabled' not in data:
        return jsonify({'error': 'Missing "enabled" boolean.'}), 400
    msg = {
        "command": "set_obstacle_motion",
        "enabled": bool(data['enabled']),
        "speed": float(data.get('speed', 0.05)),
        "velocities": data.get('velocities'),
        "bounds": data.get('bounds', {"minX": -45, "maxX": 45, "minZ": -45, "maxZ": 45}),
        "bounce": bool(data.get('bounce', True)),
    }
    ok = broadcast(msg)
    if not ok:
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'obstacle motion updated', 'config': msg})

@app.route('/collisions', methods=['GET'])
def get_collisions():
    return jsonify({'count': collision_count})

@app.route('/reset', methods=['POST', 'OPTIONS'])
def reset():
    global collision_count
    if request.method == 'OPTIONS':
        return _ok_options_response()
    collision_count = 0
    ok = broadcast({"command": "reset"})
    if not ok:
        return jsonify({'status': 'reset done (no simulators connected)', 'collisions': collision_count})
    return jsonify({'status': 'reset broadcast', 'collisions': collision_count})

# ---------------------------
# WebSocket server loop thread
# ---------------------------
def start_ws_server(host="0.0.0.0", port=8080):
    global async_loop
    loop = asyncio.new_event_loop()
    async_loop = loop

    async def _serve():
        # allow large messages (base64 images)
        server = await websockets.serve(ws_handler, host, port, max_size=None)
        log.info(f"WebSocket server listening on ws://{host}:{port}")
        return server

    def _run_loop():
        loop.run_until_complete(_serve())
        loop.run_forever()

    t = threading.Thread(target=_run_loop, name="ws-loop-thread", daemon=True)
    t.start()
    return t

# ---------------------------
# Entry: start WS thread then run Flask in main thread
# ---------------------------
if __name__ == "__main__":
    start_ws_thread = start_ws_server(host="0.0.0.0", port=8080)

    log.info("Starting Flask app on http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False, threaded=True)
