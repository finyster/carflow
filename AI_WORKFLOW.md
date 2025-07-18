# AI 協作開發工作流程指南

## 🤖 AI 模型協作規範

此文件提供給 AI 助手統一的工作流程，確保不同 AI 模型在協作開發時能保持一致性和延續性。

## 📋 必讀項目

### 1. 專案概覽
- 閱讀 `README.md` 了解專案架構
- 檢查 `config.yaml` 了解當前配置
- 查看最新的 git commit 訊息

### 2. 核心文件結構
```
carflow/
├── main.py              # 主程式 - 車輛偵測入口
├── config.yaml          # 配置檔案 - 所有參數設定
├── utils/
│   ├── tracker.py      # 追蹤器 - 核心演算法
│   ├── draw.py         # 視覺化 - 畫面標註
│   └── __init__.py     # 模組初始化
└── line_selector.py    # 工具 - 互動式線條選取
```

## 🔄 標準工作流程

### Phase 1: 任務初始化
1. **檢查當前狀態**
   ```bash
   git status
   git log --oneline -5
   ```

2. **讀取相關文件**
   - 主要程式碼檔案
   - 配置檔案
   - 相關組件文檔

3. **更新工作日誌**
   - 在本文件末尾記錄開始時間
   - 說明接收到的任務

### Phase 2: 程式碼分析
1. **理解現有架構**
   - 找出相關的類別和函數
   - 檢查依賴關係
   - 識別修改影響範圍

2. **問題定位**
   - 如果是 bug 修復，重現問題
   - 如果是新功能，找出最佳實作位置

### Phase 3: 實作階段
1. **編寫程式碼**
   - 遵循現有的程式碼風格
   - 使用既有的 import 和函式庫
   - 保持函數單一職責

2. **測試驗證**
   - 執行基本功能測試
   - 檢查是否影響其他功能

### Phase 4: 文檔更新
1. **更新組件文檔**（詳見下方模板）
2. **記錄變更內容**
3. **更新工作日誌**

## 📝 組件文檔模板

### 新增功能時使用：
```markdown
## [組件名稱] - [日期]

### 功能描述
- 主要功能：
- 輸入參數：
- 輸出結果：

### 實作細節
- 關鍵函數：`function_name()` in `file_path:line_number`
- 依賴模組：
- 演算法邏輯：

### 使用範例
```python
# 程式碼範例
```

### 已知限制
- 限制1：
- 限制2：

### 測試狀態
- [x] 基本功能
- [ ] 邊界測試
- [ ] 效能測試
```

### 修復 Bug 時使用：
```markdown
## Bug 修復 - [日期]

### 問題描述
- 錯誤現象：
- 影響範圍：
- 重現步驟：

### 根本原因
- 問題位置：`function_name()` in `file_path:line_number`
- 原因分析：

### 解決方案
- 修改內容：
- 影響範圍：

### 驗證結果
- [x] 問題已解決
- [x] 無副作用
- [x] 相關測試通過
```

## 🧠 記憶保持機制

### 上下文維護
1. **每次開始工作前**
   - 讀取此文件
   - 檢查最新的工作日誌
   - 了解上次中斷的位置

2. **工作過程中**
   - 定期更新進度
   - 記錄重要發現
   - 保存中間結果

3. **工作結束時**
   - 完整記錄所有變更
   - 更新組件狀態
   - 標註下次工作重點

### 程式碼變更追蹤
```markdown
### 最近變更記錄
| 日期 | 檔案 | 變更類型 | 描述 | 責任 AI |
|------|------|----------|------|---------|
| 2024-XX-XX | utils/tracker.py | 優化 | 改進追蹤演算法效能 | Claude |
| 2024-XX-XX | main.py | 修復 | 解決配置檔案讀取問題 | GPT-4 |
```

## ⚠️ 重要注意事項

### 安全規範
- ⚠️ **絕對禁止**：生成惡意程式碼
- ✅ **允許**：防禦性安全分析、漏洞檢測工具
- ✅ **允許**：安全文檔、檢測規則

### 檔案操作規範
- 📖 **優先讀取**：現有檔案，理解後再修改
- ✏️ **優先編輯**：現有檔案，避免創建新檔案
- 🚫 **禁止**：主動創建文檔檔案（除非明確要求）

### 程式碼品質
- 遵循 Python PEP 8 風格
- 函數需要清楚的 docstring
- 變數命名具有描述性
- 避免硬編碼，使用配置檔案

## 🔧 除錯指導原則

### 1. 問題分類
- **語法錯誤**：立即修復，檢查語法規則
- **邏輯錯誤**：分析演算法流程，添加除錯輸出
- **效能問題**：使用 profiler，找出瓶頸
- **相依性問題**：檢查 import，確認版本相容

### 2. 除錯步驟
1. 重現問題
2. 添加日誌輸出
3. 縮小問題範圍
4. 提出解決方案
5. 測試驗證
6. 清理除錯程式碼

### 3. 效能優化
- 測量前後效能差異
- 記錄優化策略
- 保留原始實作作為備份

## 📊 專案狀態總覽

### 核心組件狀態
- **main.py**: ✅ 穩定，基本功能完整
- **utils/tracker.py**: 🚧 開發中，需要效能優化
- **utils/draw.py**: ✅ 穩定，視覺化功能完整
- **config.yaml**: ⚠️ 需要擴充更多參數選項

### 目前技術債務
1. 追蹤器演算法效能問題
2. 錯誤處理機制不完整
3. 缺乏完整的單元測試

### 下一步優先級
1. **高優先級**：追蹤器效能優化
2. **中優先級**：添加錯誤處理
3. **低優先級**：補充測試覆蓋

---

## 📋 工作日誌

### 2025-01-11
**AI 助手**: Claude Sonnet 4  
**任務**: 創建 AI 協作工作流程指南  
**狀態**: ✅ 完成  
**變更**: 
- 新增 `AI_WORKFLOW.md` 檔案
- 建立統一的工作流程規範
- 添加組件文檔模板和記憶保持機制

**下次重點**: 根據實際使用情況調整工作流程

---

*此文件應保持更新，每次 AI 協作時都要參考並更新相關內容*