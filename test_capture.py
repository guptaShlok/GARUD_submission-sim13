# capture_tester.py
import requests
import json
import websockets
import asyncio
import threading
import time
from queue import Queue
import base64
import os


# 1x1 transparent PNG (small placeholder) as data URL
PLACEHOLDER_PNG = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)

class ImageCaptureTest:
    def __init__(self, api_base="http://localhost:5001", ws_url="ws://localhost:8080"):
        self.api_base = api_base
        self.ws_url = ws_url
        self.running = False
        self.messages_received = Queue()
        self.ws_connected = False
        # allow tester to act as simulator responder if no real browser client available
        self.act_as_simulator = True

    async def websocket_listener(self):
        """Listen for WebSocket messages and optionally respond to capture requests"""
        try:
            async with websockets.connect(self.ws_url, max_size=None) as websocket:
                self.ws_connected = True
                print("✅ WebSocket connected to", self.ws_url)

                while self.running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                        try:
                            data = json.loads(message)
                        except Exception:
                            data = {"raw": message}

                        print("\n📨 Received message:")
                        # print the most useful parts for debugging
                        if isinstance(data, dict):
                            print("   keys:", list(data.keys()))
                            if 'command' in data:
                                print("   command:", data['command'])
                            if 'request_id' in data:
                                print("   request_id:", data['request_id'])
                        else:
                            print("   message:", str(data)[:200])

                        # If server asked simulator(s) to capture, optionally act as simulator
                        if isinstance(data, dict) and data.get("command") == "capture_image":
                            rid = data.get("request_id")
                            print("📸 Server requested capture. request_id=", rid)
                            # form a capture response that matches your server expectations
                            response = {
                                "type": "capture_image_response",
                                "request_id": rid,
                                "image": PLACEHOLDER_PNG,
                                "timestamp": int(time.time() * 1000),
                                "position": {"x": 1, "y": 0, "z": 2}
                            }
                            # send back over websocket so server can resolve the waiting future
                            try:
                                await websocket.send(json.dumps(response))
                                print("↩️  Sent simulated capture_image_response (placeholder image).")
                            except Exception as e:
                                print("❌ Failed to send simulated capture response:", e)

                        # If we receive an actual capture response, push to queue
                        if isinstance(data, dict) and data.get("type") == "capture_image_response":
                            self.messages_received.put(data)
                            print("🎉 capture_image_response queued (len image data: {})".format(
                                len(data.get("image", "")) if data.get("image") else 0
                            ))
                        else:
                            # queue other messages for inspection
                            self.messages_received.put(data)

                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        print("WebSocket connection closed by server")
                        break
        except Exception as e:
            print("WebSocket error (listener):", e)
        finally:
            self.ws_connected = False
            print("WebSocket listener exiting")

    def start_websocket_listener(self):
        """Entry point for running the asyncio websocket listener in a thread."""
        asyncio.run(self.websocket_listener())

    def test_capture(self):
        """Test image capture functionality end-to-end."""
        print("🧪 Starting image capture test...")

        # Start WebSocket listener thread
        self.running = True
        ws_thread = threading.Thread(target=self.start_websocket_listener, daemon=True)
        ws_thread.start()

        # Wait for connection (give up after 6s)
        wait_until = time.time() + 6
        while not self.ws_connected and time.time() < wait_until:
            time.sleep(0.05)

        if not self.ws_connected:
            print("❌ WebSocket not connected. Make sure the WebSocket server is running at", self.ws_url)
            self.running = False
            return False

        # Trigger capture - use GET because server uses GET /capture
        print("\n📸 Sending capture request (HTTP GET) to", f"{self.api_base}/capture")
        try:
            response = requests.get(f"{self.api_base}/capture", timeout=12)
            print("   HTTP status:", response.status_code)
            try:
                print("   Server JSON:", response.json())
            except Exception:
                print("   Server response text:", response.text[:400])
        except Exception as e:
            print("❌ Error sending HTTP capture request:", e)
            self.running = False
            return False

        # Wait up to 10s for capture_image_response to arrive via websocket
        # --- Poll the server's /last_capture endpoint for up to 10s ---
        print("\n⏳ Polling /last_capture for server-saved image (10s)...")
        deadline = time.time() + 10
        got = False
        while time.time() < deadline:
            try:
                r = requests.get(f"{self.api_base}/last_capture", timeout=3)
            except Exception:
                time.sleep(0.25)
                continue

            if r.status_code == 200:
                d = r.json()
                # server stored the capture - we expect d to contain at least 'image' or 'type'
                img_data = d.get('image') or d.get('data') or None
                if img_data:
                    # Save PNG locally for verification
                    if isinstance(img_data, str) and img_data.startswith("data:image"):
                        header, b64 = img_data.split(',', 1)
                        try:
                            png_bytes = base64.b64decode(b64)
                        except Exception as e:
                            print("❌ Failed to decode base64 from /last_capture:", e)
                            break
                        os.makedirs("captures", exist_ok=True)
                        fname = os.path.join("captures", f"server_saved_{int(time.time()*1000)}.png")
                        with open(fname, "wb") as f:
                            f.write(png_bytes)
                        print("✅ Saved server image to", fname)
                    else:
                        print("⚠️ /last_capture returned image field but format unexpected.")
                    print("🎉 Capture confirmed via /last_capture:", d.get("timestamp"))
                    got = True
                    break
                else:
                    # If server returned metadata but no image yet, keep polling
                    print("ℹ️ /last_capture present but no image yet; retrying...")
            else:
                # 404 means server has not stored a capture yet
                # print(".", end="", flush=True)
                pass

            time.sleep(0.25)

        if not got:
            print("\n⏰ Timeout - /last_capture never returned an image.")
            # Optionally print what messages we saw on WS:
            while not self.messages_received.empty():
                print("   WS message:", self.messages_received.get())
        else:
            print("\n✅ End-to-end capture test succeeded (server-saved image).")



if __name__ == "__main__":
    tester = ImageCaptureTest(api_base="http://localhost:5001", ws_url="ws://localhost:8080")

    print("🔧 Quick health check: GET /collisions")
    try:
        r = requests.get("http://localhost:5001/collisions", timeout=3)
        if r.status_code == 200:
            print("✅ Server responded to /collisions")
        else:
            print("⚠️ /collisions returned", r.status_code, r.text)
    except Exception as e:
        print("❌ Could not contact server at http://localhost:5001 -", e)
        print("Start server.py first and ensure it's reachable.")
        raise SystemExit(1)

    ok = tester.test_capture()

    print("\n" + "=" * 40)
    if ok:
        print("✅ Tester confirms capture flow is working")
    else:
        print("❌ Capture flow failed. See logs above and check browser console (if using browser client).")
