# Auto-Driving Robot Simulation

A small autonomous robot navigation project that uses a backend Python server, a browser-based simulator, and a vision-driven navigation client (`vision_navigation.py`).  
The navigator processes camera captures (via WebSocket + HTTP), detects a colored flag (blue/cyan) and green obstacles, and issues relative move commands to reach goals while minimizing collisions.

---

## 📂 Contents

- `server.py` — backend REST & WebSocket server used by the simulator and navigator
- `simulator.html` — browser simulator that connects by WebSocket and renders the robot & scene
- `vision_navigation.py` (or `vision_navigator.py`) — the vision-based navigation client (your main robot logic)
- `captures/` — directory where captured PNGs are saved during runs
- Other helper scripts and assets (if present)

---

## ✨ Features

- WebSocket listener that receives `capture_image_response` messages (fast pipeline).
- REST endpoints for control: `/capture`, `/last_capture`, `/move_rel`, `/stop`, `/goal`, `/collisions`.
- OpenCV HSV-based detection:
  - Goal (blue/cyan flag)
  - Obstacles (green boxes)
- State-machine navigation with modes:
  - Rotating
  - Moving-to-goal
  - Safe-moving
  - Avoiding-obstacle
  - Backtracking
- Movement throttling and collision backoff.
- Saves captures to `captures/` for offline debugging.

---

## ⚙️ Requirements

- **Python** ≥ 3.8
- **Browser** (Chrome/Firefox) for the simulator
- OS: Linux / macOS / Windows

Install dependencies:

```bash
pip install numpy opencv-python requests websockets Pillow
```
