import requests
import json
import websockets
import asyncio
import threading
import time
import os
import base64
import cv2
import numpy as np
from queue import Queue, Empty
import math
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CONFIGURE HERE if needed
API_BASE = "http://localhost:5001"   # change to 5000 if server uses that
WS_URL = "ws://localhost:8080"

CAPTURE_TIMEOUT = 5           # seconds to wait for WS capture response
LAST_CAPTURE_POLL = 5        # seconds to poll /last_capture fallback
MOVE_COOLDOWN = 0.2         # seconds between move commands
COLLISIONS_POLL_INTERVAL = 5  # seconds between polling /collisions for stats

# ensure output dir
os.makedirs("captures", exist_ok=True)

# Navigation states
STATE_ROTATING = "rotating"
STATE_MOVING_TO_GOAL = "moving_to_goal"
STATE_SAFE_MOVING = "safe_moving"
STATE_BACKTRACKING = "backtracking"

def save_dataurl_png(dataurl: str, out_dir="captures"):
    """Save a data:image/png;base64,... string to disk and return filepath or None."""
    if not dataurl or not isinstance(dataurl, str):
        return None
    if "," not in dataurl:
        return None
    header, b64 = dataurl.split(",", 1)
    try:
        blob = base64.b64decode(b64)
    except Exception:
        return None
    fname = os.path.join(out_dir, f"capture_{int(time.time()*1000)}.png")
    try:
        with open(fname, "wb") as f:
            f.write(blob)
        return fname
    except Exception as e:
        logger.error(f"Failed to save PNG: {e}")
        return None


class VisionNavigation:
    def __init__(self, api_base=API_BASE, ws_url=WS_URL):
        self.api_base = api_base.rstrip("/")
        self.ws_url = ws_url
        self.running = False

        # queues for inter-thread comms
        self.ws_messages = Queue()      # all WS messages
        self.image_queue = Queue()      # capture_image_response messages
        self.event_queue = Queue()      # other events (collision, goal_reached)

        # websocket thread controls
        self._stop_ws = threading.Event()
        self._ws_thread = None
        self._ws_connected = threading.Event()

        # collision stat (server-backed but we keep a local quick counter)
        self.server_collision_count = 0
        self.local_collision_count = 0  # increments on WS collision messages
        
        # Navigation state
        self.state = STATE_ROTATING
        self.rotation_angle = 0  # Track total rotation
        self.rotation_direction = 1  # 1 for clockwise, -1 for counterclockwise
        self.last_goal_position = None
        self.consecutive_no_goal = 0
        self.max_no_goal = 3
        self.last_collision_time = 0
        self.collision_cooldown = 3.0  # seconds to wait after collision
        self.goal_corners = ["NE", "NW", "SE", "SW"]
        self.current_goal_index = 0
        self.movement_history = []  # Track recent movements for backtracking
        self.max_movement_history = 5  # Maximum number of movements to remember
        self.last_move_time = 0  # Track when we last moved
        self.goal_locked = False  # Flag to indicate if we're locked onto a goal

        # CV thresholds (HSV) - adjusted for better blue flag detection
        self.goal_lower = np.array([85, 100, 100])   # blue lower (adjusted range)
        self.goal_upper = np.array([135, 255, 255]) # blue upper (adjusted range)
        self.obstacle_lower = np.array([35, 50, 50])  # green lower
        self.obstacle_upper = np.array([85, 255, 255])# green upper

    # -----------------------
    # WebSocket listener
    # -----------------------
    async def _ws_coro(self):
        try:
            async with websockets.connect(self.ws_url, max_size=None) as ws:
                logger.info(f"WS connected: {self.ws_url}")
                self._ws_connected.set()
                while not self._stop_ws.is_set():
                    try:
                        raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    except asyncio.TimeoutError:
                        continue
                    except websockets.exceptions.ConnectionClosed:
                        logger.warning("WS connection closed by server")
                        break

                    # parse JSON
                    try:
                        msg = json.loads(raw)
                    except Exception:
                        msg = {"raw": raw}
                    # enqueue globally and classify
                    self.ws_messages.put(msg)
                    # classify common types
                    if isinstance(msg, dict):
                        t = msg.get("type")
                        if t == "capture_image_response" or "image" in msg:
                            self.image_queue.put(msg)
                        elif t == "collision" and msg.get("collision"):
                            self.event_queue.put(msg)
                            # increment local counter
                            self.local_collision_count += 1
                            self.last_collision_time = time.time()
                            logger.warning("Collision detected! Switching to backtracking state.")
                            self.state = STATE_BACKTRACKING
                            self.goal_locked = False  # Reset goal lock on collision
                        elif t == "goal_reached":
                            self.event_queue.put(msg)
                        else:
                            # fallback: if msg contains collision-like keys
                            if msg.get("collision"):
                                self.event_queue.put(msg)
                                self.local_collision_count += 1
                                self.last_collision_time = time.time()
                                logger.warning("Collision detected! Switching to backtracking state.")
                                self.state = STATE_BACKTRACKING
                                self.goal_locked = False  # Reset goal lock on collision
        except Exception as e:
            logger.error(f"WS listener error: {e}")
        finally:
            self._ws_connected.clear()

    def start_ws_thread(self):
        def run():
            asyncio.run(self._ws_coro())
        self._stop_ws.clear()
        t = threading.Thread(target=run, daemon=True, name="ws-thread")
        t.start()
        if not self._ws_connected.wait(timeout=6.0):
            logger.warning("WS did not connect within 6s. Ensure browser client is open.")
        return self._ws_connected.is_set()

    def stop_ws_thread(self):
        self._stop_ws.set()
        time.sleep(0.1)

    # -----------------------
    # REST helpers
    # -----------------------
    def rest_get(self, path, timeout=5):
        try:
            r = requests.get(f"{self.api_base}{path}", timeout=timeout)
            return r
        except Exception as e:
            logger.error(f"HTTP GET error {path}: {e}")
            return None

    def rest_post(self, path, json_payload=None, timeout=5):
        try:
            r = requests.post(f"{self.api_base}{path}", json=json_payload, timeout=timeout)
            return r
        except Exception as e:
            logger.error(f"HTTP POST error {path}: {e}")
            return None

    def capture_request(self):
        logger.debug(f"Requesting capture via {self.api_base}/capture")
        try:
            r = requests.get(f"{self.api_base}/capture", timeout=CAPTURE_TIMEOUT + 5)
            if r is None:
                logger.error("No response from capture request")
                return False, None
            try:
                info = r.json()
            except Exception:
                info = {"text": r.text}
            logger.debug(f"capture HTTP returned: {r.status_code} {info}")
            return r.status_code == 200, info
        except Exception as e:
            logger.error(f"capture_request exception: {e}")
            return False, None

    def poll_last_capture(self, timeout=LAST_CAPTURE_POLL):
        deadline = time.time() + timeout
        while time.time() < deadline:
            r = self.rest_get("/last_capture", timeout=3)
            if r is None:
                time.sleep(0.25); continue
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    data = None
                return data
            elif r.status_code == 404:
                time.sleep(0.25); continue
            else:
                logger.warning(f"/last_capture returned {r.status_code} {r.text[:200]}")
                time.sleep(0.25)
        return None

    def get_collisions_count(self):
        r = self.rest_get("/collisions", timeout=3)
        if r and r.status_code == 200:
            try:
                d = r.json()
                self.server_collision_count = int(d.get("count", 0))
            except Exception:
                pass

    # -----------------------
    # Movement helpers
    # -----------------------
    def send_move_rel(self, turn_deg=0.0, distance=0.0):
        now = time.time()
        if now - self.last_move_time < MOVE_COOLDOWN:
            if abs(turn_deg) < 1e-6 and abs(distance) < 1e-6:
                pass
            else:
                logger.debug(f"Move throttled; last move was {now - self.last_move_time:.2f}s ago")
                return False

        if abs(turn_deg) > 0.001 or abs(distance) > 0.001:
            self.movement_history.append((turn_deg, distance))
            if len(self.movement_history) > self.max_movement_history:
                self.movement_history.pop(0)
            self.last_move_time = now

        payload = {"turn": float(turn_deg), "distance": float(distance)}
        r = self.rest_post("/move_rel", json_payload=payload, timeout=5)
        if r and r.status_code == 200:
            logger.info(f"✓ Move command successful: turn={turn_deg}°, dist={distance}")
            return True
        else:
            status = r.status_code if r else "no-response"
            text = r.text[:100] if r else ""
            logger.error(f"✗ Move command failed: status={status}, text={text}")
            return False

    def send_stop(self):
        r = self.rest_post("/stop", json_payload={}, timeout=3)
        if r and r.status_code == 200:
            logger.info("✓ Stop command successful")
            return True
        logger.error("✗ Stop command failed")
        return False

    def set_goal_corner(self, corner="NE"):
        payload = {"corner": corner}
        r = self.rest_post("/goal", json_payload=payload, timeout=4)
        if r and r.status_code == 200:
            logger.info(f"✓ Goal set successfully: {r.json()}")
            return True
        logger.warning(f"✗ Failed to set goal: {(r.status_code if r else 'no-response')}")
        return False

    # -----------------------
    # Image processing (OpenCV)
    # -----------------------
    def process_image_dict(self, data):
        img_field = data.get("image") if isinstance(data, dict) else None
        if not img_field:
            logger.warning("No image field in capture response")
            return False, None, [], (0, 0), None

        if "," in img_field:
            _, b64 = img_field.split(",", 1)
        else:
            b64 = img_field
        try:
            blob = base64.b64decode(b64)
            arr = np.frombuffer(blob, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.error(f"decode image failed: {e}")
            return False, None, [], (0, 0), None

        if img is None:
            logger.error("cv2.imdecode returned None")
            return False, None, [], (0, 0), None

        h, w = img.shape[:2]
        logger.debug(f"Image shape: {img.shape}")
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        # goal detection (blue flag)
        gm = cv2.inRange(hsv, self.goal_lower, self.goal_upper)
        kernel = np.ones((3, 3), np.uint8)
        gm = cv2.morphologyEx(gm, cv2.MORPH_OPEN, kernel)
        gm = cv2.morphologyEx(gm, cv2.MORPH_CLOSE, kernel)
        
        g_cnts, _ = cv2.findContours(gm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        goal_detected = False
        goal_pos = None
        goal_area = 0
        if g_cnts:
            largest = max(g_cnts, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            logger.debug(f"Largest goal contour area: {area}")
            if area > 100:  # Minimum area threshold
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    goal_detected = True
                    goal_pos = (cx, cy)
                    goal_area = area
                    logger.info(f"✓ Goal detected at ({cx}, {cy}) with area {area}")

        # obstacles detection (green)
        om = cv2.inRange(hsv, self.obstacle_lower, self.obstacle_upper)
        om = cv2.morphologyEx(om, cv2.MORPH_OPEN, kernel)
        om = cv2.morphologyEx(om, cv2.MORPH_CLOSE, kernel)
        
        o_cnts, _ = cv2.findContours(om, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        obstacles = []
        for c in o_cnts:
            area = cv2.contourArea(c)
            if area > 500:  # Minimum area threshold
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    obstacles.append((cx, cy, int(area)))
        
        if obstacles:
            logger.debug(f"Detected {len(obstacles)} obstacles")

        return goal_detected, goal_pos, obstacles, (h, w), img

    # -----------------------
    # State machine methods
    # -----------------------
    def handle_rotating_state(self, goal_detected, goal_pos, obstacles, img_shape):
        h, w = img_shape
        
        if goal_detected and goal_pos:
            logger.info("✓ Goal detected during rotation, switching to MOVING_TO_GOAL state")
            self.state = STATE_MOVING_TO_GOAL
            self.rotation_angle = 0
            self.goal_locked = True
            return "found_goal", True
        
        turn_amount = 15 * self.rotation_direction
        self.rotation_angle += abs(turn_amount)
        
        if self.rotation_angle >= 360:
            logger.info("✓ Completed full rotation without finding goal, switching to SAFE_MOVING state")
            self.state = STATE_SAFE_MOVING
            self.rotation_angle = 0
            return "rotation_complete", True
        
        # Check for obstacles in the center region - widened for safety
        center_region_start = w * 0.3  # 30% from left
        center_region_end = w * 0.7    # 70% from left
        lower_half = h * 0.6          # Lower 60% of image
        
        obstacles_in_center = []
        for (ox, oy, area) in obstacles:
            if oy > lower_half and center_region_start <= ox < center_region_end:
                obstacles_in_center.append((ox, oy, area))
        
        if obstacles_in_center:
            self.rotation_direction *= -1
            logger.info("✓ Obstacle detected during rotation, changing direction")
            return "obstacle_avoided", True
        
        # Send rotation command
        logger.info(f"✓ Rotating: {turn_amount}°")
        self.send_move_rel(turn_amount, 0)
        return "rotating", True

    def handle_moving_to_goal_state(self, goal_detected, goal_pos, obstacles, img_shape):
        h, w = img_shape
        
        if not goal_detected or not goal_pos:
            logger.info("✓ Lost goal during movement, switching to ROTATING state")
            self.state = STATE_ROTATING
            self.rotation_angle = 0
            self.goal_locked = False
            return "lost_goal", True
        
        gx, gy = goal_pos
        
        # Calculate offset from center - FIXED: inverted the offset calculation
        offset = (w // 2) - gx  # Inverted: positive offset means goal is on the left
        normalized_offset = offset / (w // 2)  # -1 to 1
        
        # Check for obstacles in a wider path - more conservative
        # Define danger zones: center and near-center regions
        danger_zone_start = w * 0.25  # 25% from left
        danger_zone_end = w * 0.75    # 75% from left
        immediate_path = h * 0.6      # 60% of the image height
        
        obstacles_in_path = []
        for (ox, oy, area) in obstacles:
            if oy > immediate_path and danger_zone_start <= ox < danger_zone_end:
                obstacles_in_path.append((ox, oy, area))
        
        if obstacles_in_path:
            # Obstacle in path, avoid it more aggressively
            logger.info("✓ Obstacle in path, avoiding")
            # Turn away from the obstacle - FIXED: inverted the turning logic
            if gx < w // 2:
                # Goal is on left, turn left to avoid (inverted)
                turn = -30
            else:
                # Goal is on right, turn right to avoid (inverted)
                turn = 30
            
            self.send_move_rel(turn, 0)
            return "avoiding_obstacle", True
        
        # No obstacles in path, move toward goal cautiously
        # Calculate turn angle (proportional control) - FIXED: use the corrected offset
        turn = normalized_offset * 15  # Max 15 degrees for smoother movement
        
        # Calculate distance based on goal position - reduced for safety
        # If goal is in lower half (closer), move slowly
        if gy > h * 0.7:
            distance = 0.5  # Very slow when close
        elif gy > h * 0.4:
            distance = 1.0  # Medium distance
        else:
            distance = 1.5  # Far away, move faster but still cautious
        
        # Move toward goal
        logger.info(f"✓ Moving to goal: turn={turn}°, dist={distance}")
        self.send_move_rel(turn, distance)
        return "moving_to_goal", True

    def handle_safe_moving_state(self, goal_detected, goal_pos, obstacles, img_shape):
        h, w = img_shape
        
        if goal_detected and goal_pos:
            logger.info("✓ Goal detected during safe movement, switching to MOVING_TO_GOAL state")
            self.state = STATE_MOVING_TO_GOAL
            self.goal_locked = True
            return "found_goal", True
        
        # Divide the image into regions
        left_region = (0, w // 3)
        center_region = (w // 3, 2 * w // 3)
        right_region = (2 * w // 3, w)
        lower_half = h * 0.6  # Lower 60% of image
        
        obstacles_in_left = []
        obstacles_in_center = []
        obstacles_in_right = []
        
        for (ox, oy, area) in obstacles:
            if oy > lower_half:
                if left_region[0] <= ox < left_region[1]:
                    obstacles_in_left.append((ox, oy, area))
                elif center_region[0] <= ox < center_region[1]:
                    obstacles_in_center.append((ox, oy, area))
                elif right_region[0] <= ox < right_region[1]:
                    obstacles_in_right.append((ox, oy, area))
        
        # Check for obstacles in the immediate path - more conservative
        if obstacles_in_center:
            # Obstacle in center, avoid it - FIXED: inverted the turning logic
            logger.info("✓ Obstacle in center during safe movement, avoiding")
            # Turn in the direction with fewer obstacles
            if len(obstacles_in_left) <= len(obstacles_in_right):
                # Fewer obstacles on left, turn left (inverted)
                self.send_move_rel(-30, 0)
            else:
                # Fewer obstacles on right, turn right (inverted)
                self.send_move_rel(30, 0)
            return "avoiding_obstacle", True
        
        # Choose a direction with no obstacles - more cautious
        if not obstacles_in_center:
            # Center is clear, move forward cautiously
            logger.info("✓ Moving forward (center clear)")
            self.send_move_rel(0, 1.0)
            return "moving_forward", True
        elif not obstacles_in_left:
            # Left is clear, turn left and move cautiously - FIXED: inverted turn
            logger.info("✓ Moving left (left clear)")
            self.send_move_rel(-30, 0.8)
            return "moving_left", True
        elif not obstacles_in_right:
            # Right is clear, turn right and move cautiously - FIXED: inverted turn
            logger.info("✓ Moving right (right clear)")
            self.send_move_rel(30, 0.8)
            return "moving_right", True
        else:
            # All directions blocked, rotate to find a clear path
            logger.info("✓ Rotating to find clear path")
            self.send_move_rel(45, 0)
            return "rotating_to_find_path", True

    def handle_backtracking_state(self, goal_detected, goal_pos, obstacles, img_shape):
        if not self.movement_history:
            logger.info("✓ No movement history for backtracking, switching to ROTATING state")
            self.state = STATE_ROTATING
            self.rotation_angle = 0
            self.goal_locked = False
            return "no_movement_history", True
        
        # Get the last movement and reverse it
        last_turn, last_distance = self.movement_history.pop()
        
        # Reverse the movement (turn opposite direction, move same distance)
        reverse_turn = -last_turn
        reverse_distance = last_distance
        
        # Execute the reverse movement
        logger.info(f"✓ Backtracking: turn={reverse_turn}°, dist={reverse_distance}")
        self.send_move_rel(reverse_turn, reverse_distance)
        
        # After backtracking, switch to rotating state
        self.state = STATE_ROTATING
        self.rotation_angle = 0
        self.goal_locked = False
        return "backtracking_complete", True

    # -----------------------
    # Decision-making
    # -----------------------
    def decide_and_act(self, goal_detected, goal_pos, obstacles, img_shape):
        action = "idle"
        moved = False
        
        if self.state == STATE_ROTATING:
            action, moved = self.handle_rotating_state(goal_detected, goal_pos, obstacles, img_shape)
        elif self.state == STATE_MOVING_TO_GOAL:
            action, moved = self.handle_moving_to_goal_state(goal_detected, goal_pos, obstacles, img_shape)
        elif self.state == STATE_SAFE_MOVING:
            action, moved = self.handle_safe_moving_state(goal_detected, goal_pos, obstacles, img_shape)
        elif self.state == STATE_BACKTRACKING:
            action, moved = self.handle_backtracking_state(goal_detected, goal_pos, obstacles, img_shape)
        
        return action, moved

    # -----------------------
    # Main navigation loop
    # -----------------------
    def run_navigation(self):
        logger.info("Starting navigation run")
        self.running = True

        # quick server health check
        r = self.rest_get("/collisions", timeout=3)
        if not r or r.status_code != 200:
            logger.error(f"Server not available at {self.api_base}")
            return

        # start WS listener
        self.start_ws_thread()

        # set a demo goal
        self.set_goal_corner(self.goal_corners[self.current_goal_index])

        # background timer for polling collisions
        last_collisions_poll = 0.0

        try:
            while self.running:
                # poll collisions periodically
                if time.time() - last_collisions_poll > COLLISIONS_POLL_INTERVAL:
                    self.get_collisions_count()
                    logger.info(f"Collisions - local:{self.local_collision_count}  server:{self.server_collision_count}")
                    last_collisions_poll = time.time()

                # process pending events (collisions, goal reached)
                while not self.event_queue.empty():
                    evt = self.event_queue.get()
                    if isinstance(evt, dict) and evt.get("type") == "collision":
                        logger.info("✓ Collision event from WS - state already set to BACKTRACKING")
                    if isinstance(evt, dict) and evt.get("type") == "goal_reached":
                        logger.info("✓ Goal reached - setting next goal")
                        self.send_stop()
                        self.current_goal_index = (self.current_goal_index + 1) % len(self.goal_corners)
                        self.set_goal_corner(self.goal_corners[self.current_goal_index])
                        self.state = STATE_ROTATING
                        self.rotation_angle = 0
                        self.goal_locked = False

                if not self.running:
                    break

                # request capture
                ok, info = self.capture_request()
                if not ok:
                    logger.warning("✗ Capture request failed; retrying shortly")
                    time.sleep(0.5)
                    continue

                # wait for WS capture response
                got_data = None
                deadline = time.time() + CAPTURE_TIMEOUT
                while time.time() < deadline:
                    try:
                        msg = self.image_queue.get(timeout=0.3)
                    except Empty:
                        continue
                    if isinstance(msg, dict) and (msg.get("type") == "capture_image_response" or "image" in msg):
                        got_data = msg
                        break

                # fallback: poll /last_capture if no WS response
                if not got_data:
                    logger.info("ℹ No WS capture response within timeout; polling /last_capture")
                    last = self.poll_last_capture(timeout=LAST_CAPTURE_POLL)
                    if last:
                        got_data = last
                        logger.info("✓ Got last_capture fallback data")
                    else:
                        logger.warning("✗ No capture received; retrying loop")
                        continue

                # save image (if present)
                if isinstance(got_data, dict) and got_data.get("image"):
                    saved = save_dataurl_png(got_data.get("image"))
                    if saved:
                        logger.debug(f"✓ Saved capture to {saved}")

                # process image and decide
                goal_detected, goal_pos, obstacles, shape, _img = self.process_image_dict(got_data)
                if shape == (0, 0) or shape is None:
                    logger.warning("✗ Invalid image shape; skipping decision")
                    time.sleep(0.3)
                    continue

                action, moved = self.decide_and_act(goal_detected, goal_pos, obstacles, shape)
                logger.info(f"State: {self.state}, Action: {action}, Goal detected: {goal_detected}, Moved: {moved}")
                
                # short delay for motion to take effect
                time.sleep(0.5)

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            self.running = False
        finally:
            logger.info("Navigation exiting: stopping WS and robot")
            self.send_stop()
            self.stop_ws_thread()

    # -----------------------
    # Entrypoint
    # -----------------------
    def start(self):
        self.run_navigation()


if __name__ == "__main__":
    nav = VisionNavigation()
    nav.start()