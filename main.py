"""
main.py – YOLOv8 + Centroid Tracker（Route1 / Route2 車流計數）
---------------------------------------------------------------
‧ 讀取 config.yaml 的影片路徑、Entry / Exit 線
‧ YOLOv8 偵測 → 自製 Tracker（tracker.py）配唯一 ID
‧ 判斷車輛跨 Entry/Exit → routeN:序號 car 標籤
‧ 即時顯示計數、FPS，並輸出標記影片 + CSV
"""
import os, csv, time, yaml, cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from utils.tracker import Tracker   # ← 你的 tracker.py

# ---------- 幾何工具 ----------
def ccw(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

def crossed(p_prev, p_now, l1, l2):
    return ccw(p_prev, p_now, l1) != ccw(p_prev, p_now, l2)

def get_color(cls):
    pal = {
        "car": (0,255,255),
        "bus": (255,0,0),
        "truck": (0,0,255),
        "motorcycle": (0,255,0)   # 新增摩托車（綠色）
    }
    return pal.get(cls, (200,200,200))

# ---------- 讀取 config ----------
with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

routes = cfg["routes"]             # Entry / Exit 線設定
TARGET_CLASSES = cfg.get("classes",["car","bus","truck", "motorcycle"])

# ---------- YOLOv8 ----------
model = YOLO(cfg.get("model_path","yolov8s.pt"))
CLASSES = [c.strip() for c in open("coco.txt").readlines()]
CLS2ID = {c:i for i,c in enumerate(CLASSES)}

# ---------- Tracker ----------
tracker = Tracker()   # 簡易中心點追蹤

# ---------- 影片 I/O ----------
cap = cv2.VideoCapture(cfg["video_path"])
W,H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS = cap.get(cv2.CAP_PROP_FPS) or 30
os.makedirs("results",exist_ok=True)
out = cv2.VideoWriter("results/annotated.mp4",cv2.VideoWriter_fourcc(*"mp4v"),FPS,(W,H))

# ---------- 統計 ----------
route_serials = defaultdict(lambda: defaultdict(int))
vehicle_info  = {}   # id → {route,serial,class}
route_counts  = defaultdict(lambda: defaultdict(int))
last_center   = {}

# ---------- 主迴圈 ----------
frame_idx, t0 = 0, time.time()
while True:
    ret, frame = cap.read()
    if not ret: break
    frame_idx+=1
    if frame_idx % (cfg.get("skip_frame",1)+1): out.write(frame); continue

    # ---------- YOLO 偵測 ----------
    results = model(frame,verbose=False)[0]
    dets   = []  # [x1,y1,x2,y2]
    det_cls= []  # class name
    for box in results.boxes:
        cls_name = CLASSES[int(box.cls[0])]
        if cls_name not in TARGET_CLASSES: continue
        x1,y1,x2,y2 = map(int,box.xyxy[0].tolist())
        dets.append([x1,y1,x2,y2])
        det_cls.append(cls_name)

    # ---------- Tracker 更新 ----------
    # Tracker 需 (x,y,w,h)
    cv_rectangles = []
    for (x1,y1,x2,y2) in dets:
        cv_rectangles.append([x1,y1,x2-x1,y2-y1])  # 轉 w,h
    tracks = tracker.update(cv_rectangles)         # 回傳 [x,y,w,h,id]

    # ---------- 每台車處理 ----------
    # ---------- 每台車處理 ----------
    for (x, y, w, h, tid), cls_name in zip(tracks, det_cls):
        x1, y1, x2, y2 = x, y, x + w, y + h
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cur = (cx, cy)

        # 只有這台車之前出現過，才判斷是否跨線
        if tid in last_center:
            prev_center = last_center[tid]

            # STEP-1 只在進入時指定 route（不計數不給序號）
            if tid not in vehicle_info:
                for rt in routes:
                    ent1, ent2 = tuple(rt["entry"]["p1"]), tuple(rt["entry"]["p2"])
                    if crossed(prev_center, cur, ent1, ent2):
                        vehicle_info[tid] = {
                            "route": rt["name"],
                            "serial": None,  # 還沒給序號
                            "class": cls_name
                        }
                        break

            # STEP-2 只有越過出口線才 assign serial、計數
            elif vehicle_info[tid]["serial"] is None:
                rt_name = vehicle_info[tid]["route"]
                rt_cfg = next(r for r in routes if r["name"] == rt_name)
                ext1, ext2 = tuple(rt_cfg["exit"]["p1"]), tuple(rt_cfg["exit"]["p2"])
                if crossed(prev_center, cur, ext1, ext2):
                    route_serials[rt_name][cls_name] += 1
                    serial = route_serials[rt_name][cls_name]
                    vehicle_info[tid]["serial"] = serial    # 這時才給序號
                    route_counts[rt_name][cls_name] += 1    # 這時才+1計數

        # 記錄這一幀中心點（for 下一幀判斷用）
        last_center[tid] = cur


        # 畫框
        color=get_color(cls_name)
        cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)
        if tid in vehicle_info and vehicle_info[tid]["serial"]:
            label=f'{vehicle_info[tid]["route"]}:{vehicle_info[tid]["serial"]} {cls_name}'
        else:
            label=f'{cls_name} ID:{tid}'
        cv2.putText(frame,label,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

    # ---------- 畫 Entry/Exit 線 ----------
    for rt in routes:
        ent1,ent2=tuple(rt["entry"]["p1"]),tuple(rt["entry"]["p2"])
        ext1,ext2=tuple(rt["exit"]["p1"]), tuple(rt["exit"]["p2"])
        cv2.line(frame,ent1,ent2,(0,255,0),2)
        cv2.putText(frame,f'{rt["name"]}_entry',ent1,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
        cv2.line(frame,ext1,ext2,(0,0,255),2)
        cv2.putText(frame,f'{rt["name"]}_exit',ext1,cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)

    # ---------- 畫統計 ----------
    y0=40
    for i,(rt,cnts) in enumerate(route_counts.items()):
        txt=f'{rt}: '+'  '.join([f'{k}={v}' for k,v in cnts.items()])
        cv2.putText(frame,txt,(20,y0+i*30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,0),2)

    # ---------- FPS ----------
    fps=frame_idx/(time.time()-t0+1e-6)
    cv2.putText(frame,f'FPS:{fps:.1f}',(W-150,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    cv2.imshow("CarFlow",frame)
    out.write(frame)
    if cv2.waitKey(1)&0xFF==27: break

cap.release(); out.release(); cv2.destroyAllWindows()

# ---------- 輸出 CSV ----------
os.makedirs("results",exist_ok=True)
with open("results/counts.csv","w",newline="") as f:
    wr=csv.writer(f); wr.writerow(["route","class","count"])
    for rt,cnts in route_counts.items():
        for cls,c in cnts.items(): wr.writerow([rt,cls,c])
print("✅  統計寫入  results/counts.csv")
print("🎥  完成標註影片 results/annotated.mp4")
