# 模型比較實驗：同一份考卷，比較 Claude、Gemini 與地端開源模型

> Same 17-question RAG exam across cloud and local models (Claude Haiku/Sonnet, Gemini Flash-Lite, Qwen 2.5 via Ollama), comparing accuracy, refusals, validation failures, latency, and cost per 1,000 questions. Includes what went wrong along the way and a model-selection recommendation.

工作上選模型，不是選「最強的」，而是**依任務選夠用、又划算的**。這個實驗用 [course-qa-rag](https://github.com/chunyang130/course-qa-rag) 的 17 題評估（14 題應答、3 題必須拒答），讓不同模型考同一份考卷。

## 結論

**這類「依公司文件回答」的客服問答，Claude Haiku 5.5 是最佳選擇**：全部答對、最便宜、2 秒內回應。貴 20 倍的 Sonnet 5.5 在這個任務上沒有更好。

| 模型 | 應答題答對 | 拒答正確 | 每題秒數（中位數） | 每千題費用（美元） |
| --- | --- | --- | --- | --- |
| claude-haiku-5-5 | **14/14** | 3/3 | 1.86 | **0.214** |
| gemini-3.5-flash-lite | 13/14 | 3/3 | **1.01** | 0.297 |
| claude-sonnet-5-5 | 13/14 | 3/3 | 1.69 | 4.161 |
| qwen2.5:3b（地端，CPU） | 8/14 | 3/3 | 52.99 | 免 API 費用（自備運算資源） |

四個模型的重試、輸出格式錯誤、簡體字都是 0（Qwen 有 4 題觸發重試）。實際執行日期 2026-11-17，在 Google Colab 上執行；原始結果在 `results/`。

**換算成每月 1 萬題**：Haiku 約 2.1 美元、Gemini Flash-Lite 約 3.0 美元、Sonnet 約 41.6 美元。

## 依任務選模型

| 情境 | 建議 | 原因 |
| --- | --- | --- |
| 依文件回答的客服問答（本實驗） | **Claude Haiku 5.5** | 全對、最便宜；遇到需要推論的題目也能給出有條件的回答 |
| 需要最快回應、可接受偶爾不答 | Gemini 3.5 Flash-Lite | 每題約 1 秒；但遇到需要推論的題目會直接放棄 |
| 需要複雜推論的任務 | Sonnet 5.5（需另外評估） | 這次的題目太簡單，看不出它的優勢；成本是 Haiku 的 20 倍，要有明確的品質差距才值得 |
| 資料不能離開公司 | 地端開源模型，但要 GPU 與更大的模型 | 3B 模型在 CPU 上每題 53 秒，且過度拒答；需要在 GPU 上評估 7B 以上 |

## 觀察

**1. 同一題，四種行為**

q02「開課後兩週想退費可以退多少？」：退費規定是「未逾全期三分之一退 50%」，但題目沒說課程有幾週，需要推論。

| 模型 | 回答 | 行為 |
| --- | --- | --- |
| Haiku | 未逾三分之一可退 50%……資料未提供全期天數，無法判斷，建議請顧問確認 | **最理想**：說明規則，也說清楚不確定的地方 |
| Sonnet、Gemini | 資料中找不到這個問題的答案 | 過度保守 |
| Qwen 3B | 退還已繳費用的 50% | 擅自假設，沒有說明條件 |

Qwen 這題被關鍵字評分判為答對，但它漏掉了條件。這是關鍵字評分的限制：能檢查答案裡有沒有關鍵資訊，但檢查不出「少了但書」。

**2. 小模型的錯誤是「過度拒答」，不是亂答**

Qwen 3B 答錯的 6 題，全部是資料裡明明有答案，卻回答「找不到」。在企業應用中，這比亂答安全，但幾乎沒有用處。其中 2 題回答「找不到」卻又附上引用，被驗證規則擋下重試，重試後仍不正確。

**3. 貴不一定好**

Sonnet 5.5 的價格是 Haiku 的 20 倍，在這個任務上答對題數反而少一題。簡單任務用大模型，是在浪費成本。

## 過程中遇到的問題

| 問題 | 原因 | 處理 |
| --- | --- | --- |
| Claude Sonnet 5.5 回傳 400 | 這個模型不支援「強制呼叫指定工具」（tool_choice type=tool），Haiku 支援 | 程式偵測到後自動改用 auto 模式並記住，模型沒呼叫工具時從文字解析 JSON |
| Gemini 回傳 402 | API 金鑰屬於已綁定付費、但預付額度為 0 的專案，免費額度不適用 | 改在未綁定付費的新專案建立金鑰 |
| `gemini-2.5-flash-lite` 回傳 404 | 舊模型不再開放給新使用者 | 改用同等級的 `gemini-3.5-flash-lite` |
| `gemini-3.8-flash` 回傳 503 | 模型使用量過高，免費方案優先順序低 | 未列入比較 |
| `gemini-3.5-flash` 每題 84–113 秒後逾時 | 免費方案在尖峰時段排隊 | 未列入比較。**免費方案不適合需要即時回應的正式服務** |
| Colab 沒有分到 GPU | 免費方案 GPU 不足 | 開源模型改在 CPU 上執行，只跑 3B；7B 留待 GPU 可用時再測 |
| 一題逾時就中斷整個模型 | 程式只處理了伺服器錯誤，沒處理連線逾時 | 連線逾時也重試；單題失敗記為「API 失敗」並繼續 |
| Qwen 回答「不可以」被判錯 | 評分關鍵字漏了「不可以」 | 補上關鍵字；評分可從已存的回答重算，不必重新呼叫模型 |

## 公平比較的設計

- **相同檢索結果**：所有模型拿到完全相同的 4 個檢索段落，只比較「生成」這一步，差異才是模型本身造成的
- **相同驗證規則**：引用必須是段落原文、答案裡的數字必須出現在引用段落，不合格就回饋重試一次
- **相同評分**：應答題要有關鍵資訊且通過驗證；拒答題必須回答「資料中找不到」
- **費用**：依官網牌價（2026 年 11 月）與實際 token 用量換算；Gemini 免費額度也以付費牌價換算，方便比較

| 模型 | 價格（每百萬 token，輸入／輸出） |
| --- | --- |
| claude-haiku-5-5 | $0.10／$0.50 |
| claude-sonnet-5-5 | $2／$10 |
| gemini-3.5-flash-lite | $0.30／$2.50 |
| qwen2.5:3b | 免 API 費用，需自備運算資源 |

## 限制

- 題目只有 17 題、每個模型只跑一次，結論適用於「依公司文件回答」這類任務，不代表模型的整體能力
- 速度在 Colab（美國機房）測得，會受網路位置與當下負載影響
- 地端模型在 CPU 上執行，速度不代表 GPU 環境的表現
- Haiku 是在加入 auto 模式之前跑的，使用強制呼叫工具；Sonnet 使用 auto 模式

## 執行方式

**Colab（建議）**：打開 `benchmark_colab.ipynb`，在 Secrets 放入 API 金鑰後，由上而下逐格執行。

**本機**：

```bash
python bench.py --provider mock                                   # 免費示範
python bench.py --provider claude --model claude-haiku-5-5        # 需 ANTHROPIC_API_KEY
python bench.py --provider gemini --model gemini-3.5-flash-lite   # 需 GEMINI_API_KEY
python bench.py --provider ollama --model qwen2.5:3b              # 需 Ollama
python bench.py --report                                          # 重新評分並產生比較表
```

只用 Python 標準函式庫（3.10 以上）。

## 檔案說明

| 檔案 | 內容 |
| --- | --- |
| `bench.py` | 三家模型的呼叫、驗證、評分、重新評分、成本計算與比較表 |
| `rag.py` | 檢索與驗證（沿用 course-qa-rag），含 Claude 工具呼叫的自動降級 |
| `benchmark_colab.ipynb` | Colab 執行筆記本 |
| `results/` | 實際執行的原始結果與比較表 |
| `eval/questions.json` | 17 題評估題組 |
| `knowledge_base/` | 虛構的補習班文件 |
