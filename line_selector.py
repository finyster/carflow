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

# --- 參數設定 ---
parser = argparse.ArgumentParser(description="Select entry/exit lines for each route")
parser.add_argument("--video", required=True)
parser.add_argument("--config", default="config.yml")
args = parser.parse_args()

# --- 讀取影片第一幀 ---
cap = cv2.VideoCapture(args.video)
ret, frame = cap.read()
if not ret:
    raise RuntimeError("❌ 無法讀取影片")

# --- 全域變數 ---
clicks = []
routes = []
route_idx = 1
line_type = "entry"  # 切換 entry / exit

def mouse_cb(event, x, y, flags, _):
    global clicks, line_type, route_idx, frame
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1) # 在點擊處畫一個黃色小點

        # 每兩點連成一線
        if len(clicks) == 2:
            p1, p2 = clicks
            
            # 根據是 entry 或 exit 決定線的顏色
            color = (0, 255, 0) if line_type == "entry" else (0, 0, 255)
            name = f"route{route_idx}_{line_type}"
            cv2.line(frame, p1, p2, color, 2)
            cv2.putText(frame, f"{name}_{line_type}", p1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

            # --- ✨【核心修改】✨ ---
            if line_type == "entry":
                # 建立新路線時，除了 entry 座標外，一併加入 name 欄位
                routes.append({
                    "name": name, 
                    "entry": {"p1": list(p1), "p2": list(p2)}
                })
                line_type = "exit" # 下一條線是 exit
            else:
                # 將 exit 座標加入到最後一筆（也就是剛建立的）路線中
                routes[-1]["exit"] = {"p1": list(p1), "p2": list(p2)}
                line_type = "entry" # 下一條線是新的 entry
                route_idx += 1      # 路線編號 +1
            
            clicks = [] # 清空點擊記錄

# --- 主程式 ---
cv2.namedWindow("Select Entry/Exit Lines")
cv2.setMouseCallback("Select Entry/Exit Lines", mouse_cb)
print("👉 每『兩點』成一線，先畫 entry（進入口），再畫 exit（出口），自動一組，想多畫幾組都可以！")
print("👉 按 s 儲存，r 重畫，Esc 不存檔離開。")

# Real-time preview loop: listen for user key commands.
while True:
    cv2.imshow("Select Entry/Exit Lines", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('r'): # 按 r 重設
        frame = original_frame.copy() # 還原成最原始的畫面
        routes.clear()
        clicks.clear()
        route_idx = 1
        line_type = "entry"
        print("🔄 重設完成，重新畫線")
    elif key == ord('s'):
        break

    elif key == 27: # 按 Esc 離開
        cap.release()
        cv2.destroyAllWindows()
        print("⚠️ 取消儲存，離開")
        exit(0)

cap.release()
cv2.destroyAllWindows()

# ---------- 寫入 config.yml ----------
yaml_path = Path(args.config)

# 如果設定檔存在，先讀取既有內容
if yaml_path.exists():
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
else:
    cfg = {}

# 更新或新增 video_path 和 routes
cfg["video_path"] = args.video
cfg["routes"] = routes

# 將更新後的內容寫回檔案
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)

print(f"✅ 已寫入 {yaml_path}，總共 {len(routes)} 組 route (每組 entry/exit)")
