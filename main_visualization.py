# main_visualization.py – v13.2 (儲存車種資訊)
import os, time, yaml, cv2, json
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm

def get_color(cls):
    pal = {"car": (0, 255, 255), "bus": (255, 0, 0), "truck": (0, 0, 255), "motorcycle": (0, 255, 0)}
    return pal.get(cls, (200, 200, 200))

def main():
    try:
        with open("config.yaml") as f: cfg = yaml.safe_load(f)
    except FileNotFoundError: print("❌ 錯誤：找不到 config.yaml 設定檔！"); return

    video_path = cfg.get("video_path")
    model_path = cfg.get("model_path", "yolov8s.pt")
    output_video_path = cfg.get("save_video", "results/annotated_tracks.mp4")
    target_classes = cfg.get("classes", ["car", "bus", "truck", "motorcycle"])
    
    print("初始化模型..."); model = YOLO(model_path)
    tracker = DeepSort(max_age=50, n_init=3, nms_max_overlap=1.0)

    cap = cv2.VideoCapture(video_path)
    W, H, FPS = (int(cap.get(p)) for p in [cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS])
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS or 30, (W, H))

    # ✨【核心修改 1】✨ 建立一個新的資料結構來儲存車種和軌跡
    vehicle_tracks = defaultdict(lambda: {"class": None, "track": []})
    
    print(f"🚀 開始繪製並收集軌跡: {video_path}")
    
    try:
        for _ in tqdm(range(total_frames), desc="影片處理進度"):
            ret, frame = cap.read()
            if not ret: break

            results = model(frame, verbose=False, conf=0.3, imgsz=1280)
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
                
                # ✨【核心修改 2】✨ 將車種和軌跡點同時存入新的資料結構
                info = vehicle_tracks[track_id]
                if info["class"] is None:
                    info["class"] = cls_name
                info["track"].append((cx, cy))
                
                color = get_color(cls_name)
                label = f'{cls_name} ID:{track_id}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                pts = np.array(info["track"], np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)
            
            out.write(frame)

    finally:
        print("資源釋放與存檔..."); cap.release(); out.release(); cv2.destroyAllWindows()
        print(f"🎉 軌跡繪製完成！影片已儲存至: {output_video_path}")

        # ✨【核心修改 3】✨ 將新的、更豐富的資料結構寫入 JSON 檔案
        trajectory_file_path = "results/trajectories.json"
        with open(trajectory_file_path, 'w') as f:
            json.dump(vehicle_tracks, f, indent=4)
        print(f"📈 帶有車種資訊的完整軌跡數據已儲存至: {trajectory_file_path}")

if __name__ == "__main__":
    main()