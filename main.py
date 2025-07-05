# nightMain.py – 最終修正版 v2
import os, csv, time, yaml, cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
# 確保您的 tracker.py 檔案在 utils 資料夾中
from utils.tracker import Tracker

# ---------- 幾何工具 (無變動) ----------
def ccw(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

def crossed(p_prev, p_now, l1, l2):
    return ccw(p_prev, p_now, l1) != ccw(p_prev, p_now, l2)

def get_color(cls):
    pal = {
        "car": (0,255,255), "bus": (255,0,0),
        "truck": (0,0,255), "motorcycle": (0,255,0)
    }
    return pal.get(cls, (200,200,200))

# ---------- 主要執行區塊 ----------
def main():
    # 1. 讀取設定檔
    try:
        with open("night_config.yaml") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("❌ 錯誤：找不到 night_config.yaml 設定檔！")
        return

    # 從設定檔讀取參數
    video_path = cfg.get("video_path")
    model_path = cfg.get("model_path", "yolov8s.pt")
    output_video_path = cfg.get("save_video", "results/output.mp4")
    output_csv_path = cfg.get("save_csv", "results/counts.csv")
    target_classes = cfg.get("classes", ["car", "bus", "truck", "motorcycle"])
    skip_frame = cfg.get("skip_frame", 1)
    routes = cfg.get("routes", [])

    # 2. 初始化模型與工具
    print("初始化模型...")
    model = YOLO(model_path)
    try:
        with open("coco.txt") as f:
            CLASSES = [c.strip() for c in f.readlines()]
    except FileNotFoundError:
        print("❌ 錯誤：找不到 coco.txt 類別檔！")
        return
    tracker = Tracker()

    # 3. 初始化影片讀取與寫入
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 錯誤：無法開啟影片檔案：{video_path}")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs("results", exist_ok=True)
    out = cv2.VideoWriter(output_video_path, fourcc, FPS, (W, H))

    if not out.isOpened():
        print(f"❌ 錯誤：無法初始化影片寫入器。請檢查您的 OpenCV/FFmpeg 安裝。")
        cap.release()
        return

    # 4. 初始化統計變數
    vehicle_info, last_center = {}, {}
    route_counts = defaultdict(lambda: defaultdict(int))
    route_serials = defaultdict(lambda: defaultdict(int))
    frame_idx, t0 = 0, time.time()

    print(f"🚀 開始處理影片：{video_path}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n影片處理完成。")
                break
            frame_idx += 1

            if frame_idx % 100 == 0:
                elapsed_time = time.time() - t0
                fps_estimate = frame_idx / elapsed_time if elapsed_time > 0 else 0
                print(f"  處理中... 第 {frame_idx} 幀 (當前速度: {fps_estimate:.1f} FPS)")

            # AI 偵測與追蹤邏輯
            results = model(frame, verbose=False)[0]
            dets, det_cls = [], []
            for box in results.boxes:
                cls_name = CLASSES[int(box.cls[0])]
                if cls_name in target_classes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    dets.append([x1, y1, x2, y2])
                    det_cls.append(cls_name)

            rects = [[x, y, x2-x, y2-y] for (x, y, x2, y2) in dets]
            tracks = tracker.update(rects)

            # 跨線計數與繪圖邏輯
            for (x, y, w, h, tid), cls_name in zip(tracks, det_cls):
                cx, cy = x + w // 2, y + h // 2
                if tid in last_center:
                    prev_center = last_center[tid]
                    if tid not in vehicle_info:
                        for rt_cfg in routes:
                            pt1 = (int(rt_cfg["entry"]["p1"][0]), int(rt_cfg["entry"]["p1"][1]))
                            pt2 = (int(rt_cfg["entry"]["p2"][0]), int(rt_cfg["entry"]["p2"][1]))
                            if crossed(prev_center, (cx, cy), pt1, pt2):
                                vehicle_info[tid] = {"route": rt_cfg["name"], "serial": None, "class": cls_name}
                                break
                    elif vehicle_info[tid]["serial"] is None:
                        rt_name = vehicle_info[tid]["route"]
                        rt_cfg = next((r for r in routes if r["name"] == rt_name), None)
                        if rt_cfg:
                            pt1 = (int(rt_cfg["exit"]["p1"][0]), int(rt_cfg["exit"]["p1"][1]))
                            pt2 = (int(rt_cfg["exit"]["p2"][0]), int(rt_cfg["exit"]["p2"][1]))
                            if crossed(prev_center, (cx, cy), pt1, pt2):
                                route_serials[rt_name][cls_name] += 1
                                serial = route_serials[rt_name][cls_name]
                                vehicle_info[tid]["serial"] = serial
                                route_counts[rt_name][cls_name] += 1
                last_center[tid] = (cx, cy)

                # 繪製標籤與框線
                color = get_color(cls_name)
                label = f'{cls_name} ID:{tid}'
                if tid in vehicle_info and vehicle_info[tid]["serial"]:
                    label = f'{vehicle_info[tid]["route"]}:{vehicle_info[tid]["serial"]} {cls_name}'
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 繪製路線與統計數據
            for rt_cfg in routes:
                entry_p1 = (int(rt_cfg["entry"]["p1"][0]), int(rt_cfg["entry"]["p1"][1]))
                entry_p2 = (int(rt_cfg["entry"]["p2"][0]), int(rt_cfg["entry"]["p2"][1]))
                exit_p1 = (int(rt_cfg["exit"]["p1"][0]), int(rt_cfg["exit"]["p1"][1]))
                exit_p2 = (int(rt_cfg["exit"]["p2"][0]), int(rt_cfg["exit"]["p2"][1]))
                cv2.line(frame, entry_p1, entry_p2, (0, 255, 0), 2)
                cv2.line(frame, exit_p1, exit_p2, (0, 0, 255), 2)

            y0 = 40
            # 【修正】這裡的 rt 就是路線名稱 (字串)，不應該用 rt["name"]
            for i, (rt_name, cnts) in enumerate(route_counts.items()):
                txt = f'{rt_name}: ' + ' | '.join([f'{k}={v}' for k, v in cnts.items()])
                cv2.putText(frame, txt, (20, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            out.write(frame)

    except Exception as e:
        import traceback
        print(f"\n❌ 處理過程中發生未預期的錯誤：{e}")
        print(traceback.format_exc())
    finally:
        print("釋放資源...")
        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"寫入統計資料至 {output_csv_path}...")
        with open(output_csv_path, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["route", "class", "count"])
            for rt_name, cnts in route_counts.items():
                for cls, c in cnts.items():
                    wr.writerow([rt_name, cls, c])
        
        total_time = time.time() - t0
        print("\n🎉 全部任務完成！")
        print(f"📄  統計 CSV 已儲存：{output_csv_path}")
        print(f"🎥  標註影片已儲存：{output_video_path}")
        print(f"⏱️  總耗時：{total_time:.2f} 秒")

if __name__ == "__main__":
    main()