# plot_matplotlib.py (v2 - 帶有圖例與方向指示)
import json
import argparse
import yaml
import matplotlib.pyplot as plt
import cv2 # 需要 cv2 來讀取影片尺寸

# --- 輔助函式 ---
def get_color(cls):
    """根據車輛類別回傳一個固定的顏色"""
    pal = {
        "car": "gold",         # 黃色
        "bus": "blue",         # 藍色
        "truck": "red",          # 紅色
        "motorcycle": "lime"   # 綠色
    }
    return pal.get(cls, "gray") # 其他用灰色

def main(track_file, config_file, output_image):
    # --- 載入軌跡數據 ---
    try:
        with open(track_file, 'r') as f:
            trajectories = json.load(f)
    except FileNotFoundError:
        print(f"❌ 錯誤：找不到軌跡數據檔案！路徑: {track_file}"); return
        
    # --- 從 config.yaml 獲取影片尺寸，用於設定圖表大小 ---
    try:
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        video_path = cfg.get("video_path")
        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
    except Exception as e:
        print(f"⚠️ 警告：讀取影片尺寸失敗 ({e})，將使用預設圖表大小。")
        W, H = 1920, 1080 # 預設值

    print("🎨 開始使用 Matplotlib 繪製軌跡圖表 (含圖例與起終點)...")
    
    fig, ax = plt.subplots(figsize=(W / 100, H / 100))

    # ✨【核心修改 1】✨ 使用字典來追蹤已經繪製過的類別，避免圖例重複
    handles = {}

    for track_id, info in trajectories.items():
        vehicle_class = info.get("class")
        track_points = info.get("track")
        
        # 軌跡至少要有 5 個點，方向才比較穩定
        if not vehicle_class or not track_points or len(track_points) < 5:
            continue
        
        x_coords, y_coords = zip(*track_points)
        color = get_color(vehicle_class)
        
        # 繪製軌跡線。只為每個類別的第一條線加上 label，用於生成圖例
        if vehicle_class not in handles:
            line, = ax.plot(x_coords, y_coords, color=color, linewidth=1.5, label=vehicle_class)
            handles[vehicle_class] = line
        else:
            ax.plot(x_coords, y_coords, color=color, linewidth=1.5)

        # ✨【核心修改 2】✨ 標記起始點和終點
        # 1. 標記起始點 (用一個較大的圓點)
        start_point = track_points[0]
        ax.plot(start_point[0], start_point[1], marker='o', markersize=5, color=color, alpha=0.8)

        # 2. 標記終點 (用一個小箭頭)
        end_point = track_points[-1]
        prev_point = track_points[-5] # 取倒數第5個點，讓箭頭方向更穩定
        dx = end_point[0] - prev_point[0]
        dy = end_point[1] - prev_point[1]
        ax.arrow(end_point[0] - dx*0.1, end_point[1] - dy*0.1, dx*0.1, dy*0.1, 
                 head_width=15, head_length=20, fc=color, ec=color, length_includes_head=True)


    # --- 美化圖表 ---
    ax.set_title("Vehicle Trajectories (2D Plot)")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.invert_yaxis()
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal', adjustable='box')
    
    # ✨【核心修改 3】✨ 顯示圖例
    # 根據收集到的 handles 來建立圖例
    ax.legend(handles=handles.values())

    plt.savefig(output_image, dpi=150)
    print(f"✅ 軌跡圖表繪製完成！已儲存至: {output_image}")
    
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="將車輛軌跡繪製成 Matplotlib 2D 圖表")
    parser.add_argument("--tracks", default="results/trajectories.json", help="儲存軌跡的 JSON 檔案路徑")
    parser.add_argument("--config", default="config.yaml", help="用於讀取影片尺寸的設定檔")
    parser.add_argument("--output", default="results/trajectory_matplotlib_plot.png", help="輸出的軌跡圖表檔名 (建議用 .png)")
    
    args = parser.parse_args()
    main(args.tracks, args.config, args.output)