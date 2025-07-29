# plot_trajectories.py (v4.1 - 互動式 + 方向指示優化)
import cv2
import json
import numpy as np
import argparse
import yaml

# --- 全域變數 ---
current_filter = None
redraw_flag = True

# --- 輔助函式 ---
def get_color(cls):
    pal = {"car": (0, 255, 255), "bus": (255, 0, 0), "truck": (0, 0, 255), "motorcycle": (0, 255, 0)}
    return pal.get(cls, (200, 200, 200))

def mouse_callback(event, x, y, flags, param):
    """處理滑鼠點擊事件，檢查是否點擊在按鈕上"""
    global current_filter, redraw_flag
    buttons = param['buttons']
    if event == cv2.EVENT_LBUTTONDOWN:
        for btn_name, rect in buttons.items():
            if rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
                if btn_name == 'All':
                    current_filter = None
                    print("✅ 已選擇顯示 [全部] 車種")
                else:
                    current_filter = [btn_name]
                    print(f"✅ 已選擇只顯示 [{btn_name}]")
                redraw_flag = True
                break

def main(track_file, background_mode, output_image, video_path=None):
    # --- 載入背景圖片 ---
    bg_img = None
    if background_mode in ['video', 'black', 'white']:
        if not video_path: print("❌ 錯誤：使用 'video', 'black', 或 'white' 模式時，必須提供 --video 參數！"); return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): print(f"❌ 錯誤：無法開啟影片檔案！路徑: {video_path}"); return
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if background_mode == 'video':
            ret, frame = cap.read(); bg_img = frame
            if not ret: print("❌ 錯誤：無法讀取影片的第一幀！"); cap.release(); return
        else:
            color = (0, 0, 0) if background_mode == 'black' else (255, 255, 255)
            bg_img = np.full((H, W, 3), color, dtype=np.uint8)
        cap.release()
    else:
        try: bg_img = cv2.imread(background_mode); assert bg_img is not None
        except: print(f"❌ 錯誤：找不到指定的背景圖片檔案！路徑: {background_mode}"); return

    # --- 載入軌跡數據 ---
    try:
        with open(track_file, 'r') as f: trajectories = json.load(f)
    except FileNotFoundError: print(f"❌ 錯誤：找不到軌跡數據檔案！路徑: {track_file}"); return

    # --- 建立 UI 按鈕 ---
    all_classes = sorted(list(set(info['class'] for info in trajectories.values() if info.get('class'))))
    button_names = ['All'] + all_classes
    buttons = {}
    btn_x, btn_y, btn_w, btn_h = 20, 20, 150, 40
    for i, name in enumerate(button_names):
        x1 = btn_x + i * (btn_w + 10)
        y1 = btn_y
        buttons[name] = (x1, y1, x1 + btn_w, y1 + btn_h)

    # --- 主迴圈與事件處理 ---
    window_name = "Interactive Trajectory Analyzer"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback, {'buttons': buttons})

    print("\n--- 互動操作說明 ---")
    print("👉 在上方選擇車種按鈕，即可篩選顯示的軌跡。")
    print("👉 按 's' 鍵儲存目前的畫面。")
    print("👉 按 'q' 鍵或 Esc 離開。")

    global redraw_flag
    while True:
        if redraw_flag:
            display_frame = bg_img.copy()

            for name, rect in buttons.items():
                color = (0, 255, 0) if (current_filter is None and name == 'All') or (current_filter and name in current_filter) else (80, 80, 80)
                cv2.rectangle(display_frame, (rect[0], rect[1]), (rect[2], rect[3]), color, -1)
                cv2.putText(display_frame, name, (rect[0] + 10, rect[3] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
            plot_count = 0
            for track_id, info in trajectories.items():
                vehicle_class = info.get("class")
                track_points = info.get("track")
                if not vehicle_class or not track_points or len(track_points) < 5: continue # 建議軌跡至少有5個點，讓箭頭方向更穩定

                if current_filter is None or vehicle_class in current_filter:
                    color = get_color(vehicle_class)
                    pts = np.array(track_points, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(display_frame, [pts], isClosed=False, color=color, thickness=2)

                    # 繪製起點圓圈
                    start_point = tuple(track_points[0])
                    cv2.circle(display_frame, start_point, 7, color, -1)
                    
                    # ✨【核心修改】✨ 讓終點箭頭更清晰可見
                    # 為了讓箭頭方向更穩定，我們取軌跡的最後幾個點來計算
                    end_point = tuple(track_points[-1])
                    prev_point = tuple(track_points[-5]) # 取倒數第5個點，讓箭頭的向量更長、方向更準

                    # 增加線條粗細 (4)，並增大箭頭比例 (0.5)
                    cv2.arrowedLine(display_frame, prev_point, end_point, color, 4, tipLength=0.5)

                    plot_count += 1
            
            redraw_flag = False

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(20) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('s'):
            cv2.imwrite(output_image, display_frame)
            print(f"🖼️ 畫面已儲存至: {output_image}")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="互動式軌跡分析儀")
    parser.add_argument("--tracks", default="results/trajectories_complex-2.json", help="儲存軌跡的 JSON 檔案路徑")
    parser.add_argument("--background", default="video", help="背景模式: 'video', 'black', 'white', 或圖片檔案路徑")
    parser.add_argument("--output", default="results/interactive_plot.jpg", help="按下 's' 鍵時，儲存的圖片檔名")
    parser.add_argument("--video", help="原始影片路徑 (使用 'video', 'black', 'white' 模式時需要)")
    
    args = parser.parse_args()
    
    if args.background in ['video', 'black', 'white'] and not args.video:
        try:
            with open("config.yaml") as f: cfg = yaml.safe_load(f)
            args.video = cfg.get("video_path")
            if not args.video: raise ValueError("config.yaml 中未找到 video_path")
            print(f"ℹ️ 已從 config.yaml 自動讀取影片路徑: {args.video}")
        except Exception as e: print(f"❌ 錯誤：必須提供 --video 參數。 ({e})"); exit()
    
    main(args.tracks, args.background, args.output, args.video)