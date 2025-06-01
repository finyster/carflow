# main.py — CarFlow project (可快速互動＋完整跑完並輸出)
import os, csv, time, yaml, cv2, numpy as np
from collections import defaultdict
from ultralytics import YOLO
from utils.tracker import Tracker

# ---------- Geometry helpers ----------
def ccw(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

def crossed(p_prev, p_now, l1, l2):
    return ccw(p_prev, p_now, l1) != ccw(p_prev, p_now, l2)

def get_color(cls):
    pal = {
        "car": (0,255,255),
        "bus": (255,0,0),
        "truck": (0,0,255),
        "motorcycle": (0,255,0)
    }
    return pal.get(cls, (200,200,200))

# ---------- 讀取配置 ----------
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

routes = cfg["routes"]
TARGET_CLASSES = cfg.get("classes", ["car","bus","truck","motorcycle"])
MODEL_PATH = cfg.get("model_path", "yolov8l.pt")

# ---------- 建立 YOLOv8 model, Centroid Tracker 等全域物件 ----------
model   = YOLO(MODEL_PATH)
CLASSES = [c.strip() for c in open("coco.txt").readlines()]

# process_full_video() 跟 preview_and_wait() 共用的欄位
WIDTH  = None
HEIGHT = None
FPS    = None

# ---------- 定義：完整跑完影片、寫出 CSV 的函式 ----------
def process_full_video(video_path: str, out_video_path: str = "results/annotated_full.mp4"):
    """
    這個函式不開啟任何視窗，直接在背景一幀一幀跑完整影片，
    結束後再把 route/class/count 統計寫入 CSV。

    如果 you want the annotated video as well, it 會寫到 out_video_path。
    """
    global WIDTH, HEIGHT, FPS

    # 1) 開影片與初始化輸出影片（annotated_full.mp4）
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法打開影片檔：{video_path}")

    WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS    = cap.get(cv2.CAP_PROP_FPS) or 30

    os.makedirs("results", exist_ok=True)
    out = cv2.VideoWriter(
        out_video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (WIDTH, HEIGHT)
    )

    # 2) 建立 Tracker、計數結構
    tracker        = Tracker()
    route_serials  = defaultdict(lambda: defaultdict(int))
    vehicle_info   = {}
    route_counts   = defaultdict(lambda: defaultdict(int))
    last_center    = {}

    frame_idx, t0 = 0, time.time()
    skip = cfg.get("skip_frame", 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 如果想跳過某些幀，加速計算，可取消下面註解
        # if frame_idx % (skip + 1) != 1:
        #     out.write(frame)
        #     continue

        # ---------- YOLO 偵測 ----------
        results = model(frame, verbose=False)[0]
        dets, det_cls = [], []
        for box in results.boxes:
            cls_name = CLASSES[int(box.cls[0])]
            if cls_name not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append([x1, y1, x2, y2])
            det_cls.append(cls_name)

        # ---------- 更新 Tracker ----------
        rects  = [[x1, y1, x2-x1, y2-y1] for (x1, y1, x2, y2) in dets]
        tracks = tracker.update(rects)

        # ---------- 線交叉檢測 & 計數邏輯 ----------
        for (x, y, w, h, tid), cls_name in zip(tracks, det_cls):
            x1, y1, x2, y2 = x, y, x + w, y + h
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            cur = (cx, cy)

            if tid in last_center:
                prev = last_center[tid]

                # STEP1：cross entry → assign route，但暫不計數
                if tid not in vehicle_info:
                    for rt in routes:
                        ent1, ent2 = tuple(rt["entry"]["p1"]), tuple(rt["entry"]["p2"])
                        if crossed(prev, cur, ent1, ent2):
                            vehicle_info[tid] = {
                                "route": rt["name"],
                                "serial": None,
                                "class": cls_name
                            }
                            break

                # STEP2：cross exit → assign serial 並計數
                elif vehicle_info[tid]["serial"] is None:
                    rt_name = vehicle_info[tid]["route"]
                    rt_cfg  = next(r for r in routes if r["name"] == rt_name)
                    ext1, ext2 = tuple(rt_cfg["exit"]["p1"]), tuple(rt_cfg["exit"]["p2"])
                    if crossed(prev, cur, ext1, ext2):
                        route_serials[rt_name][cls_name] += 1
                        serial = route_serials[rt_name][cls_name]
                        vehicle_info[tid]["serial"] = serial
                        route_counts[rt_name][cls_name] += 1

            last_center[tid] = cur

            # 畫框與標籤
            color = get_color(cls_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if tid in vehicle_info and vehicle_info[tid]["serial"]:
                label = f'{vehicle_info[tid]["route"]}:{vehicle_info[tid]["serial"]} {cls_name}'
            else:
                label = f'{cls_name} ID:{tid}'
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # ---------- 畫 entry/exit 線與統計（同 preview） ----------
        for rt in routes:
            e1, e2 = tuple(rt["entry"]["p1"]), tuple(rt["entry"]["p2"])
            x1, x2 = tuple(rt["exit"]["p1"]), tuple(rt["exit"]["p2"])
            cv2.line(frame, e1, e2, (0, 255, 0), 2)
            cv2.putText(frame, f'{rt["name"]}_entry', e1,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.line(frame, x1, x2, (0, 0, 255), 2)
            cv2.putText(frame, f'{rt["name"]}_exit', x1,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        y0 = 40
        for i, (rt, cnts) in enumerate(route_counts.items()):
            txt = f'{rt}: ' + ' | '.join([f'{k}={v}' for k, v in cnts.items()])
            cv2.putText(frame, txt, (20, y0 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        fps = frame_idx / (time.time() - t0 + 1e-6)
        cv2.putText(frame, f'FPS:{fps:.1f}', (WIDTH-150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        out.write(frame)

    cap.release()
    out.release()

    # ---------- 寫 CSV Summary ----------
    os.makedirs("results", exist_ok=True)
    with open("results/counts_full.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["route", "class", "count"])
        for rt, cnts in route_counts.items():
            for cls, c in cnts.items():
                wr.writerow([rt, cls, c])

    print("Full-process complete. CSV → results/counts_full.csv")
    print(f"Annotated video → {out_video_path}")



# ---------- 定義：互動式預覽函式 ==========
def preview_and_wait(video_path: str):
    """
    這段函式只做「帶有即時顯示」的互動式 preview。
    可以用 skip_frame、或按快捷鍵 (f / Esc / q) 來快速略過或結束預覽。
    預覽結束後才回到 main，真正的「完整跑流程」會放在 process_full_video()。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"無法打開影片檔：{video_path}")

    WIDTH  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS    = cap.get(cv2.CAP_PROP_FPS) or 30

    # 建立可縮放視窗，並稍微把它設成 800×600
    cv2.namedWindow("Preview", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Preview", 800, 600)

    tracker        = Tracker()
    route_serials  = defaultdict(lambda: defaultdict(int))
    vehicle_info   = {}
    route_counts   = defaultdict(lambda: defaultdict(int))
    last_center    = {}

    frame_idx, t0 = 0, time.time()
    skip = cfg.get("skip_frame", 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 按 'f' 快轉 5 幀
        key = cv2.waitKey(1) & 0xFF
        if key == ord('f'):
            for _ in range(5):
                if not cap.read()[0]:
                    break
            frame_idx += 5
            continue
        # 按 'q' 或 'Esc' 提前結束 preview
        if key == ord('q') or key == 27:
            break

        # skip_frame：只在每隔 skip 幀時做偵測、畫框，其餘直接顯示
        if frame_idx % (skip + 1) != 1:
            cv2.imshow("Preview", frame)
            continue

        # 以下偵測、追蹤、畫框、畫統計，跟 background 類似，只是顯示出來
        results = model(frame, verbose=False)[0]
        dets, det_cls = [], []
        for box in results.boxes:
            cls_name = CLASSES[int(box.cls[0])]
            if cls_name not in TARGET_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            dets.append([x1, y1, x2, y2])
            det_cls.append(cls_name)

        rects  = [[x1, y1, x2-x1, y2-y1] for (x1, y1, x2, y2) in dets]
        tracks = tracker.update(rects)

        for (x, y, w, h, tid), cls_name in zip(tracks, det_cls):
            x1, y1, x2, y2 = x, y, x + w, y + h
            cx, cy = (x1 + x2)//2, (y1 + y2)//2
            cur = (cx, cy)

            if tid in last_center:
                prev = last_center[tid]

                if tid not in vehicle_info:
                    for rt in routes:
                        ent1, ent2 = tuple(rt["entry"]["p1"]), tuple(rt["entry"]["p2"])
                        if crossed(prev, cur, ent1, ent2):
                            vehicle_info[tid] = {
                                "route": rt["name"], "serial": None, "class": cls_name
                            }
                            break
                elif vehicle_info[tid]["serial"] is None:
                    rt_name = vehicle_info[tid]["route"]
                    rt_cfg  = next(r for r in routes if r["name"] == rt_name)
                    ext1, ext2 = tuple(rt_cfg["exit"]["p1"]), tuple(rt_cfg["exit"]["p2"])
                    if crossed(prev, cur, ext1, ext2):
                        route_serials[rt_name][cls_name] += 1
                        serial = route_serials[rt_name][cls_name]
                        vehicle_info[tid]["serial"] = serial
                        route_counts[rt_name][cls_name] += 1
            last_center[tid] = cur

            color = get_color(cls_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if tid in vehicle_info and vehicle_info[tid]["serial"]:
                label = f'{vehicle_info[tid]["route"]}:{vehicle_info[tid]["serial"]} {cls_name}'
            else:
                label = f'{cls_name} ID:{tid}'
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 畫 entry/exit 線與統計
        for rt in routes:
            e1, e2 = tuple(rt["entry"]["p1"]), tuple(rt["entry"]["p2"])
            x1, x2 = tuple(rt["exit"]["p1"]), tuple(rt["exit"]["p2"])
            cv2.line(frame, e1, e2, (0, 255, 0), 2)
            cv2.putText(frame, f'{rt["name"]}_entry', e1,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.line(frame, x1, x2, (0, 0, 255), 2)
            cv2.putText(frame, f'{rt["name"]}_exit', x1,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        y0 = 40
        for i, (rt, cnts) in enumerate(route_counts.items()):
            txt = f'{rt}: ' + ' | '.join([f'{k}={v}' for k, v in cnts.items()])
            cv2.putText(frame, txt, (20, y0 + i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        fps = frame_idx / (time.time() - t0 + 1e-6)
        cv2.putText(frame, f'FPS:{fps:.1f}', (WIDTH-150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Preview", frame)

    cap.release()
    cv2.destroyAllWindows()

    print("Preview finished. Press Enter to start full-process counting···")
    input()  # 等使用者按下 Enter 才繼續


# ---------- main() 入口 ==========
if __name__ == "__main__":
    video_path = cfg["video_path"]
    # 1) 先做互動式的 preview（可以快轉、快速 Skip）
    preview_and_wait(video_path)

    # 2) Preview 完後，做「完整跑完影片且寫 CSV」的 background 流程
    process_full_video(video_path, out_video_path="results/annotated_full.mp4")
