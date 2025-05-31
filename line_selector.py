# line_selector_routes.py
import cv2
import yaml
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description="Select entry/exit lines for each route")
parser.add_argument("--video", required=True)
parser.add_argument("--config", default="config.yml")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("❌ 無法讀取影片")

clicks = []
routes = []
route_idx = 1
line_type = "entry"  # 切換 entry / exit

def mouse_cb(event, x, y, flags, _):
    global clicks, line_type, route_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        if len(clicks) == 2:
            p1, p2 = clicks
            color = (0, 255, 0) if line_type == "entry" else (0, 0, 255)
            name = f"route{route_idx}_{line_type}"
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
print("👉 每『兩點』成一線，先畫 entry（進入口），再畫 exit（出口），自動一組，想多畫幾組都可以！")
print("👉 按 s 儲存，r 重畫，Esc 不存檔離開。")

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
        print("🔄 重設完成，重新畫線")
    elif key == ord('s'):
        break
    elif key == 27:
        cap.release()
        cv2.destroyAllWindows()
        print("⚠️ 取消儲存，離開")
        exit(0)

cap.release()
cv2.destroyAllWindows()

# ---------- 寫入 config.yml ----------
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

print(f"✅ 已寫入 {yaml_path}，總共 {len(routes)} 組 route (每組 entry/exit)")
