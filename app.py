import streamlit as st
import cv2
import yaml
import os
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from utils.tracker import Tracker # 確保 tracker.py 在 utils 資料夾中
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import subprocess

# --- 核心功能函式化 (從您的 main.py 和 utils 改寫) ---

def get_color(cls):
    pal = {
        "car": (255, 191, 0), "bus": (255, 64, 0),
        "truck": (0, 128, 255), "motorcycle": (0, 224, 128)
    }
    return pal.get(cls, (200, 200, 200))

def ccw(a, b, c):
    return (c[1]-a[1])*(b[0]-a[0]) > (b[1]-a[1])*(c[0]-a[0])

def crossed(p_prev, p_now, l1, l2):
    return ccw(p_prev, p_now, l1) != ccw(p_prev, p_now, l2)

# 主要的影片處理函式
def process_video(video_path, routes_config, progress_bar, status_text):
    # 1. 初始化模型與工具
    model = YOLO("yolov8n.pt") # 使用較小的模型以提升網頁反應速度
    try:
        with open("coco.txt") as f:
            CLASSES = [c.strip() for c in f.readlines()]
    except FileNotFoundError:
        st.error("錯誤：找不到 coco.txt 類別檔！")
        return None, None
    tracker = Tracker()
    target_classes = ["car", "bus", "truck", "motorcycle"]

    # 2. 初始化影片讀取與寫入
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"錯誤：無法開啟影片檔案：{video_path}")
        return None, None

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 確保 results 資料夾存在
    os.makedirs("results", exist_ok=True)
    output_video_path = f"results/output_{int(time.time())}.mp4"
    output_csv_path = f"results/counts_{int(time.time())}.csv"
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, FPS, (W, H))

    # 3. 初始化統計變數
    vehicle_info, last_center = {}, {}
    route_counts = defaultdict(lambda: defaultdict(int))
    route_serials = defaultdict(lambda: defaultdict(int))
    frame_idx, t0 = 0, time.time()

    # 4. 主迴圈
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 更新進度條
        progress = frame_idx / total_frames
        elapsed_time = time.time() - t0
        eta = (total_frames - frame_idx) * (elapsed_time / frame_idx) if frame_idx > 0 else 0
        progress_bar.progress(progress)
        status_text.text(f"處理中... {int(progress * 100)}% (預計剩餘時間: {int(eta)} 秒)")

        # AI 偵測與追蹤
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

        # 跨線計數與繪圖
        for (x, y, w, h, tid), cls_name in zip(tracks, det_cls):
            cx, cy = x + w // 2, y + h // 2
            if tid in last_center:
                prev_center = last_center[tid]
                # 檢查是否進入某個路線
                if tid not in vehicle_info:
                    for rt_cfg in routes_config:
                        pt1, pt2 = rt_cfg["entry"]["p1"], rt_cfg["entry"]["p2"]
                        if crossed(prev_center, (cx, cy), pt1, pt2):
                            vehicle_info[tid] = {"route": rt_cfg["name"], "serial": None, "class": cls_name}
                            break
                # 如果已進入路線，檢查是否離開
                elif vehicle_info[tid]["serial"] is None:
                    rt_name = vehicle_info[tid]["route"]
                    rt_cfg = next((r for r in routes_config if r["name"] == rt_name), None)
                    if rt_cfg:
                        pt1, pt2 = rt_cfg["exit"]["p1"], rt_cfg["exit"]["p2"]
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
        for rt_cfg in routes_config:
            cv2.line(frame, tuple(rt_cfg["entry"]["p1"]), tuple(rt_cfg["entry"]["p2"]), (0, 255, 0), 2)
            cv2.line(frame, tuple(rt_cfg["exit"]["p1"]), tuple(rt_cfg["exit"]["p2"]), (0, 0, 255), 2)

        y0 = 40
        for i, (rt_name, cnts) in enumerate(route_counts.items()):
            txt = f'{rt_name}: ' + ' | '.join([f'{k}={v}' for k, v in cnts.items()])
            cv2.putText(frame, txt, (20, y0 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        out.write(frame)

    # 5. 釋放資源並儲存CSV
    cap.release()
    out.release()
    status_text.text("處理完成！")

    df = pd.DataFrame([(r, c, cnt) for r, cl in route_counts.items() for c, cnt in cl.items()], columns=["route", "class", "count"])
    df.to_csv(output_csv_path, index=False)

    return output_video_path, output_csv_path

# --- Streamlit UI ---

# 初始化 session state
if 'upload_history' not in st.session_state:
    st.session_state.upload_history = []
if 'routes' not in st.session_state:
    st.session_state.routes = []
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'processed_video' not in st.session_state:
    st.session_state.processed_video = None
if 'processed_csv' not in st.session_state:
    st.session_state.processed_csv = None

st.set_page_config(page_title="CarFlow 智慧車流分析", layout="wide")
st.title("🚗 CarFlow 智慧車流分析系統")
st.info("一個整合 YOLOv8 與 Streamlit 的互動式車輛偵測與計數平台。")


# --- 側邊欄 ---
with st.sidebar:
    st.header("⚙️ 控制面板")

    # 1. 影片上傳
    uploaded_file = st.file_uploader("1. 上傳您的影片檔案", type=["mp4", "mov", "avi"])
    if uploaded_file:
        # 將上傳的檔案存到暫存区
        temp_video_path = f"temp_{uploaded_file.name}"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 如果是新影片，重置狀態
        if st.session_state.video_path != temp_video_path:
            st.session_state.video_path = temp_video_path
            st.session_state.routes = []
            st.session_state.processed_video = None
            st.session_state.processed_csv = None
            st.session_state.upload_history.append({"name": uploaded_file.name, "date": time.strftime("%Y-%m-%d %H:%M:%S")})
            st.rerun() # 使用最新版的 rerun

    # 2. 檔案轉換工具
    with st.expander("🔄 影片格式轉換工具 (.h264 → .mp4)"):
        h264_file = st.file_uploader("上傳 .h264 檔案", type=['h264'])
        if h264_file:
            if st.button("開始轉換"):
                input_path = f"temp_{h264_file.name}"
                output_path = f"converted_{os.path.splitext(h264_file.name)[0]}.mp4"
                with open(input_path, 'wb') as f:
                    f.write(h264_file.getbuffer())
                
                with st.spinner("正在使用 ffmpeg 轉換中..."):
                    try:
                        subprocess.run(['ffmpeg', '-y', '-i', input_path, '-c:v', 'copy', output_path], check=True, capture_output=True)
                        st.success(f"轉換成功！")
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="下載轉換後的 MP4",
                                data=file,
                                file_name=output_path,
                                mime="video/mp4"
                            )
                        os.remove(input_path)
                    except (subprocess.CalledProcessError, FileNotFoundError) as e:
                        st.error("轉換失敗！請確認您的系統已安裝 ffmpeg 並將其加入環境變數中。")
                        st.code(f"錯誤訊息: {e}")

    # 3. 上傳歷史
    with st.expander("🗂️ 上傳歷史紀錄"):
        if st.session_state.upload_history:
            for record in reversed(st.session_state.upload_history):
                st.write(f"📁 {record['name']} ({record['date']})")
        else:
            st.write("尚未有上傳紀錄。")

# --- 主畫面 ---
if st.session_state.video_path:
    col1, col2 = st.columns(2)

    with col1:
        st.header("🎨 步驟 1: 繪製偵測路線")
        st.write("請用滑鼠畫出車流的 **進入(綠)** 與 **離開(紅)** 線。每兩點連成一線。")
        
        cap = cv2.VideoCapture(st.session_state.video_path)
        ret, bg_frame = cap.read()
        cap.release()
        
        if ret:
            bg_image = Image.fromarray(cv2.cvtColor(bg_frame, cv2.COLOR_BGR2RGB))
            
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=3,
                stroke_color="#00FF00",
                background_image=bg_image,
                update_streamlit=True,
                height=bg_image.height,
                width=bg_image.width,
                drawing_mode="line",
                key="canvas",
            )
            
            line_type = "進入" if (not canvas_result.json_data or not canvas_result.json_data["objects"] or len(canvas_result.json_data["objects"]) % 2 == 0) else "離開"
            
            # 使用 st.markdown 來顯示帶有顏色的標題
            st.markdown(f"#### 現在請繪製: **{line_type}** 線 (顏色: {'<span style=\"color:green\">綠色</span>' if line_type == '進入' else '<span style=\"color:red\">紅色</span>'})", unsafe_allow_html=True)


            if st.button("清除重畫"):
                 st.session_state.routes = []
                 st.rerun() # 使用最新版的 rerun

            if canvas_result.json_data is not None and canvas_result.json_data["objects"]:
                st.session_state.routes = []
                lines = canvas_result.json_data["objects"]
                for i in range(0, len(lines), 2):
                    if i+1 < len(lines):
                        route_name = f"路線{i//2 + 1}"
                        entry_line = lines[i]
                        exit_line = lines[i+1]
                        st.session_state.routes.append({
                            "name": route_name,
                            "entry": {"p1": [int(entry_line['left']), int(entry_line['top'])], "p2": [int(entry_line['x2']), int(entry_line['y2'])]},
                            "exit": {"p1": [int(exit_line['left']), int(exit_line['top'])], "p2": [int(exit_line['x2']), int(exit_line['y2'])]}
                        })
                if st.session_state.routes:
                    st.write("✅ 已定義路線:", st.session_state.routes)


        st.header("🚀 步驟 2: 開始分析")
        if st.button("開始分析車流", disabled=not st.session_state.routes):
            progress_bar = st.progress(0)
            status_text = st.empty()
            with st.spinner('模型正在全力運轉中...'):
                processed_video, processed_csv = process_video(
                    st.session_state.video_path, 
                    st.session_state.routes,
                    progress_bar,
                    status_text
                )
                st.session_state.processed_video = processed_video
                st.session_state.processed_csv = processed_csv
            st.success("影片分析完成！請至右側查看結果。")

    with col2:
        st.header("📊 分析結果")
        if st.session_state.processed_video:
            st.subheader("📈 車流計數數據")
            try:
                df = pd.read_csv(st.session_state.processed_csv)
                st.dataframe(df)
                with open(st.session_state.processed_csv, "rb") as file:
                    st.download_button("下載數據 (CSV)", file, file_name=os.path.basename(st.session_state.processed_csv))
            except FileNotFoundError:
                st.warning("找不到數據檔案。")

            st.subheader("🎥 標註後影片")
            try:
                with open(st.session_state.processed_video, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
                    st.download_button("下載標註影片", video_bytes, file_name=os.path.basename(st.session_state.processed_video))
            except FileNotFoundError:
                st.warning("找不到影片檔案。")
        else:
            st.info("分析完成後，結果將會顯示在此處。")

else:
    st.info("👈 請從左側側邊欄開始，上傳一個影片檔案。")