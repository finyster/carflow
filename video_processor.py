# video_processor.py
import subprocess
import argparse
import sys
from shutil import which

def check_ffmpeg():
    """檢查系統是否安裝了 ffmpeg"""
    if which("ffmpeg") is None:
        print("❌ 錯誤：找不到 ffmpeg。")
        print("請先安裝 ffmpeg，並確保它在您的系統 PATH 中。")
        print(" - Ubuntu/Debian: sudo apt install ffmpeg")
        print(" - macOS (Homebrew): brew install ffmpeg")
        print(" - Windows: 從官網下載後設定環境變數。")
        sys.exit(1)
    print("✅ FFmpeg 已安裝。")

def process_video(input_path, output_path, duration=None):
    """
    使用 ffmpeg 轉檔並選擇性地剪輯影片。
    - 轉為 MP4 (H.264 影像, AAC 音訊)
    - 如果提供了 duration，則剪輯指定長度
    """
    if duration:
        print(f"🚀 開始處理影片：{input_path} (剪輯前 {duration} 秒)")
    else:
        print(f"🚀 開始處理影片：{input_path} (處理整部影片)")
    
    # --- ✨【核心修改】✨ ---
    # 動態建立指令列表
    command = [
        'ffmpeg',
        '-i', input_path,
    ]
    
    # 只有在 duration 有被指定時，才加入剪輯相關的參數
    if duration:
        command.extend(['-t', str(duration)])
    
    # 加入轉檔與覆蓋的參數
    command.extend([
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-y', output_path
    ])
    
    try:
        print(f"🔄 執行指令：{' '.join(command)}")
        # 執行指令，並捕獲輸出
        result = subprocess.run(
            command, 
            check=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8', # 明確指定編碼
            errors='ignore'   # 忽略潛在的解碼錯誤
        )
        print("🎉 影片處理完成！")
        print(f"📄 輸出檔案已儲存至：{output_path}")

    except FileNotFoundError:
        print(f"❌ 錯誤：找不到輸入檔案 {input_path}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        # 如果 ffmpeg 執行失敗，印出錯誤訊息
        print("❌ 錯誤：ffmpeg 處理過程中發生錯誤。")
        print("--- FFmpeg 錯誤訊息 ---")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    # 檢查 ffmpeg 是否存在
    check_ffmpeg()
    
    # 設定命令列參數
    parser = argparse.ArgumentParser(
        description="影片轉檔與剪輯工具。",
        formatter_class=argparse.RawTextHelpFormatter # 讓 help 訊息格式更好看
    )
    parser.add_argument("--input", required=True, help="來源影片的路徑")
    parser.add_argument("--output", required=True, help="輸出影片的路徑")
    parser.add_argument(
        "--duration", 
        type=int, 
        help="（可選）要剪輯的影片長度（秒）。\n若不提供此參數，則會處理整部影片。"
    )
    
    args = parser.parse_args()
    
    process_video(args.input, args.output, args.duration)