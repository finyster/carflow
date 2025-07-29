# main.py – v10.4 (穩定版：DeepSORT + 多邊形 + 雙重計數 + 右對齊修正)
import os, csv, time, yaml, cv2
import numpy as np
from collections import defaultdict, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

# --- 輔助函式 ---
def is_inside_polygon(point, zone_points):
    return cv2.pointPolygonTest(np.array(zone_points, np.int32), point, False) >= 0

def get_color(cls):
    pal = {"car": (0, 255, 255), "bus": (255, 0, 0), "truck": (0, 0, 255), "motorcycle": (0, 255, 0)}
    return pal.get(cls, (200, 200, 200))

# ---------- 主要執行區塊 ----------
def main():
    try:
        with open("config.yaml") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ 錯誤：找不到 config.yaml 設定檔！"); return

    video_path = cfg.get("video_path")
    model_path = cfg.get("model_path", "yolov8s.pt")
    output_video_path = cfg.get("save_video", "results/annotated.mp4")
    route_csv_path = cfg.get("save_csv", "results/route_counts.csv")
    zone_csv_path = "results/zone_traffic.csv"

    target_classes = cfg.get("classes", ["car", "bus", "truck", "motorcycle"])
    zones = cfg.get("zones", [])

    print("初始化模型..."); model = YOLO(model_path)
    tracker = DeepSort(max_age=50, n_init=3, nms_max_overlap=1.0)

    cap = cv2.VideoCapture(video_path)
    W, H, FPS = (int(cap.get(p)) for p in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS or 30, (W, H))

    vehicle_info = {}
    route_counts = defaultdict(lambda: defaultdict(int))
    zone_counts = defaultdict(lambda: defaultdict(int))
    track_history = defaultdict(lambda: deque(maxlen=50))
    
    print(f"🚀 開始處理影片（雙重計數模式）: {video_path}")
    try:
        for _ in tqdm(range(total_frames), desc="影片處理進度"):
            ret, frame = cap.read()
            if not ret: break

            results = model(frame, verbose=False, conf=0.2, imgsz=1280)
            
            detections = []
            for box in results[0].boxes:
                cls_name = model.names[int(box.cls[0])]
                if cls_name in target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls_name))

            tracks = tracker.update_tracks(detections, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed() or track.time_since_update > 1: continue
                track_id = track.track_id; cls_name = track.get_det_class(); ltrb = track.to_ltrb()
                x1, y1, x2, y2 = map(int, ltrb)
                cx, cy = (x1 + x2) // 2, y2
                track_history[track_id].append((cx, cy))
                
                current_zone = next((z['name'] for z in zones if is_inside_polygon((cx, cy), z['points'])), None)

                if current_zone:
                    if track_id not in vehicle_info:
                        vehicle_info[track_id] = {"last_zone": current_zone, "class": cls_name}
                        zone_counts[current_zone][cls_name] += 1
                    else:
                        info = vehicle_info[track_id]
                        if current_zone != info['last_zone']:
                            route_name = f"{info['last_zone']}_to_{current_zone}"
                            route_counts[route_name][cls_name] += 1
                            zone_counts[current_zone][cls_name] += 1
                            info['last_zone'] = current_zone
                
                color = get_color(cls_name)
                label = f'{cls_name} ID:{track_id}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                for i in range(1, len(track_history[track_id])):
                    if track_history[track_id][i-1] is None or track_history[track_id][i] is None: continue
                    cv2.line(frame, track_history[track_id][i-1], track_history[track_id][i], color, 2)
            
            for z in zones:
                pts = np.array(z['points'], np.int32)
                cv2.polylines(frame, [pts], True, (255, 255, 0), 2)
                text_pos = tuple(pts[0])
                cv2.putText(frame, z['name'], (text_pos[0], text_pos[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
            
            y0 = 40
            for i, (rt,c) in enumerate(route_counts.items()):
                s = ", ".join([f"{cls}={n}" for cls,n in c.items()])
                cv2.putText(frame, f'{rt}: {s}', (20,y0+i*30), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)

            # ✨【核心修改】✨ 繪製單一 Zone 總流量計數 (改為右對齊)
            y0_right = 40
            right_margin = 20 # 離螢幕右邊緣的距離
            font_face = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2

            # 繪製標題 "Zone Traffic"
            title_text = "Zone Traffic"
            (title_w, title_h), _ = cv2.getTextSize(title_text, font_face, 0.8, 2)
            cv2.putText(frame, title_text, (W - right_margin - title_w, y0_right), font_face, 0.8, (255,0,255), 2)
            y0_right += 35

            # 繪製每一條 Zone 的計數
            for i, (zn, c) in enumerate(zone_counts.items()):
                s = ", ".join([f"{cls}={n}" for cls, n in c.items()])
                text_line = f'{zn}: {s}'
                
                # 計算文字的寬度
                (text_w, text_h), _ = cv2.getTextSize(text_line, font_face, font_scale, font_thickness)
                
                # 計算新的 x 座標，使其右對齊
                text_x = W - right_margin - text_w
                text_y = y0_right + i * 30

                # 使用新的座標來繪製文字
                cv2.putText(frame, text_line, (text_x, text_y), font_face, font_scale, (255, 0, 255), font_thickness)
            
            out.write(frame)

    finally:
        print("資源釋放與存檔..."); cap.release(); out.release(); cv2.destroyAllWindows()
        with open(route_csv_path, "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["route", "class", "count"])
            for r, c in route_counts.items():
                for cls, n in c.items(): wr.writerow([r, cls, n])
        print(f"✅ 路徑計數已儲存至: {route_csv_path}")

        with open(zone_csv_path, "w", newline="") as f:
            wr = csv.writer(f); wr.writerow(["zone", "class", "count"])
            for z, c in zone_counts.items():
                for cls, n in c.items(): wr.writerow([z, cls, n])
        print(f"✅ 區域總流量已儲存至: {zone_csv_path}")
        
        print("\n🎉 全部任務完成！")

if __name__ == "__main__":
    main()