# CarFlow

這個專案示範如何使用 YOLOv8 與自製 Centroid Tracker 進行車輛偵測與計數：
- **main.py**：主程式，讀取 `config.yaml` 設定，執行偵測與畫面標註後輸出。
- **config.yaml**：可設定影片路徑、偵測類別、entry/exit 線等參數。
- **utils/**：
  - **draw.py**：負責在影像上畫出線條、框線、計數等資訊。
  - **tracker.py**：簡易中心點追蹤器。
- **line_selector.py**：輔助繪製 entry/exit 線的工具。
- **results/**：儲存輸出影片與計數 CSV。
- **videos/**：預設放置輸入影片的資料夾。

## 使用方式
1. 調整 `config.yaml` 以設定影片檔案及 routes。
2. 執行：
   ```bash
   python main.py
   ```
3. 執行完成後，結果會輸出至 `results/annotated.mp4` 與 `results/counts.csv`。

## 額外工具
- `line_selector.py` 可用於互動式選取 entry/exit 線：
  ```bash
  python line_selector.py --video <路徑>
  ```

歡迎根據需求修改或擴充此專案。