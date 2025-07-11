# line_selector_routes.py
import cv2
import yaml
import argparse
from pathlib import Path

"""
line_selector_routes.py
-----------------------
Interactively define traffic‑counting routes for an intersection video.

• Left‑click two points to create one line.  
  ‑ **Entry line (green)** is drawn first,  
  ‑ **Exit line (red)** follows automatically.  
  Each entry/exit pair constitutes one route.
• Press **r** to reset and redraw, **s** to save, **Esc** to quit without saving.
• The script writes/updates `config.yml` with keys:
    video_path: <input video>
    routes:
      - entry: {p1: [x, y], p2: [x, y]}
        exit : {p1: [x, y], p2: [x, y]}

Author : Lin J-H and Hjc , 2025
"""

parser = argparse.ArgumentParser(description="Select entry/exit lines for each route")
parser.add_argument("--video", required=True)
parser.add_argument("--config", default="config.yaml")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Unable to read video file.")

clicks = []
routes = []
route_idx = 1
line_type = "entry"  # toggles between drawing entry (green) and exit (red) lines

# Mouse callback: collect two clicks → draw line, store coordinates.
def mouse_cb(event, x, y, flags, _):
    global clicks, line_type, route_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        if len(clicks) == 2:
            p1, p2 = clicks
            color = (0, 255, 0) if line_type == "entry" else (0, 0, 255)
            name = f"route{route_idx}_{line_type}"
            # Draw the line on the preview frame and label it.
            cv2.line(frame, p1, p2, color, 2)
            cv2.putText(frame, name, p1, cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2, cv2.LINE_AA)
            if line_type == "entry":
                routes.append({"entry": {"p1": list(p1), "p2": list(p2)}})
                line_type = "exit"
            else:
                routes[-1]["exit"] = {"p1": list(p1), "p2": list(p2)}
                line_type = "entry"
                route_idx += 1
            clicks = []

cv2.namedWindow("Select Entry/Exit Lines")
cv2.setMouseCallback("Select Entry/Exit Lines", mouse_cb)
print("Click TWO points to make ONE line. Draw an entry line first (green), then an exit line (red). Repeat for as many routes as needed.")
print("Press 's' to save, 'r' to reset and redraw, or Esc to exit without saving.")

# Real-time preview loop: listen for user key commands.
while True:
    cv2.imshow("Select Entry/Exit Lines", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        frame[:] = frame.copy()
        ret, frame = cap.read(0)
        routes.clear()
        clicks.clear()
        route_idx = 1
        line_type = "entry"
        print("Reset complete. You can start drawing again.")
    elif key == ord('s'):
        break
    elif key == 27:
        cap.release()
        cv2.destroyAllWindows()
        print("Exit without saving.")
        exit(0)

cap.release()
cv2.destroyAllWindows()

# Save all collected routes back to config.yml
yaml_path = Path(args.config)
if yaml_path.exists():
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {}

cfg["video_path"] = args.video
cfg["routes"] = routes

with open(yaml_path, "w") as f:
    yaml.dump(cfg, f, sort_keys=False, allow_unicode=True)

print(f"Wrote {yaml_path}. Total routes saved: {len(routes)} (each includes an entry and exit line).")
