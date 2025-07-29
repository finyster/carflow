# quad_selector.py
import cv2
import yaml
import argparse
import numpy as np
from pathlib import Path

# --- 全域變數 ---
zones = []
current_points = []
zone_idx = 1
frame_copy = None

def mouse_cb(event, x, y, flags, param):
    """滑鼠回呼函式，用於定義四邊形的四個頂點"""
    global current_points, zones, zone_idx, frame

    if event == cv2.EVENT_LBUTTONDOWN:
        # 每點擊一次，新增一個頂點
        current_points.append((x, y))
        
        # 在畫面上畫出點和連接線
        cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)
        if len(current_points) > 1:
            cv2.line(frame, current_points[-2], current_points[-1], (255, 255, 0), 2)
        cv2.imshow("Select Quadrilateral Zones", frame)

        # --- ✨【核心修改】✨ 當集滿四個點時，自動完成區域 ---
        if len(current_points) == 4:
            zone_name = f"Z{zone_idx}"
            
            # 將四個點連成封閉的四邊形
            pts = np.array(current_points, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, zone_name, current_points[0], cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, (255, 255, 255), 2)
            
            # 儲存區域資訊
            zones.append({"name": zone_name, "points": [list(p) for p in current_points]})
            print(f"✅ 已新增區域: {zone_name}")
            
            # 重設，準備下一個四邊形
            current_points = []
            zone_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="互動式選取四邊形區域 (Quadrilateral Zones)")
    parser.add_argument("--video", required=True, help="來源影片路徑")
    parser.add_argument("--config", default="config.yaml", help="設定檔路徑")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    ret, frame = cap.read()
    if not ret: raise RuntimeError("❌ 無法讀取影片")
    frame_copy = frame.copy()
    
    cv2.namedWindow("Select Quadrilateral Zones")
    cv2.setMouseCallback("Select Quadrilateral Zones", mouse_cb)

    print("👉 請依序用「左鍵」點擊四個點，以定義一個四邊形區域。")
    print("👉 完成一個區域後，可繼續定義下一個。")
    print("👉 按 's' 儲存 | 按 'r' 重畫 | 按 Esc 不存檔離開")

    while True:
        cv2.imshow("Select Quadrilateral Zones", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'): break
        elif key == ord('r'):
            frame = frame_copy.copy()
            zones.clear()
            current_points.clear()
            zone_idx = 1
            print("🔄 已重設，請重新繪製所有區域。")
        elif key == 27:
            print("⚠️ 操作取消，未儲存。")
            exit()
            
    cv2.destroyAllWindows()

    yaml_path = Path(args.config)
    cfg = {}
    if yaml_path.exists():
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    
    cfg['zones'] = zones
        
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, sort_keys=False, allow_unicode=True, default_flow_style=False)
    
    print(f"\n✅ 成功將 {len(zones)} 個區域儲存至 {yaml_path}")