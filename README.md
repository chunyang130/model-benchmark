# 模型比較實驗：同一份考卷，比較 Claude、Gemini 與地端開源模型

> Same 17-question RAG exam across cloud and local models (Claude, Gemini, Qwen via Ollama), comparing accuracy, refusals, validation failures, Traditional-Chinese compliance, latency, and cost per 1,000 questions.

工作上選模型，不是「選最強的」，而是**依任務選夠用、又划算的**。這個實驗用 [course-qa-rag](https://github.com/chunyang130/course-qa-rag) 的 17 題評估（14 題應答、3 題必須拒答），讓不同模型考同一份考卷。

**實際執行結果：測試中，完成後補上。**

## 公平比較的設計

- **相同檢索結果**：所有模型拿到完全相同的 4 個檢索段落，只比較「生成」這一步，差異才是模型本身造成的
- **相同驗證規則**：引用必須是段落原文、答案裡的數字必須出現在引用段落，不合格就回饋重試一次（每個模型規則相同）
- **相同評分**：應答題要有關鍵資訊且通過驗證；拒答題必須回答「資料中找不到」

## 比較項目

| 項目 | 為什麼重要 |
| --- | --- |
| 應答題答對、拒答正確 | 品質 |
| 觸發重試、重試後仍未通過驗證 | 引用與數字是否可靠 |
| 首次輸出格式錯誤 | 能不能穩定輸出 JSON，影響能否接進系統 |
| 使用簡體字 | 題目要求繁體中文，開源小模型常改用簡體 |
| 每題秒數（中位數） | 使用者等待時間 |
| 每千題費用（美元） | 依官網牌價與實際 token 用量換算 |

## 參賽模型

| 模型 | 類型 | 價格（每百萬 token，輸入／輸出） |
| --- | --- | --- |
| claude-haiku-5-5 | 雲端，小 | $0.10／$0.50 |
| claude-sonnet-5-5 | 雲端，中 | $2／$10 |
| gemini-2.5-flash-lite | 雲端，小 | $0.10／$0.40 |
| gemini-3.8-flash | 雲端，中 | $0.75／$3.75 |
| qwen2.5:3b | 地端開源（Ollama） | 免 API 費用，需自備 GPU |
| qwen2.5:7b | 地端開源（Ollama） | 免 API 費用，需自備 GPU |

價格為 2026 年 11 月查詢的官網牌價，以各家官網為準。Gemini 免費方案的費用以付費牌價換算，方便比較。

## 執行方式

**Colab（建議）**：打開 `benchmark_colab.ipynb`，選 T4 GPU、在 Secrets 放入 API 金鑰後，由上而下執行。地端模型在 Colab 的 GPU 上執行，自己的電腦不需安裝任何東西。

**本機**：

```bash
python bench.py --provider mock                                   # 免費示範
python bench.py --provider claude --model claude-haiku-5-5        # 需 ANTHROPIC_API_KEY
python bench.py --provider gemini --model gemini-2.5-flash-lite   # 需 GEMINI_API_KEY
python bench.py --provider ollama --model qwen2.5:3b              # 需本機或遠端 Ollama
python bench.py --report                                          # 產生比較表
```

只用 Python 標準函式庫（3.10 以上）。

## 檔案說明

| 檔案 | 內容 |
| --- | --- |
| `bench.py` | 三家模型的呼叫、驗證、評分、成本計算與比較表 |
| `rag.py` | 檢索與驗證（沿用 course-qa-rag） |
| `benchmark_colab.ipynb` | Colab 執行筆記本 |
| `eval/questions.json` | 17 題評估題組 |
| `knowledge_base/` | 虛構的補習班文件 |
