# 設計文件：結構化任務回報 + 獨立 Reviewer Agent 機制

> 版本：v0.2 — 2026-06-27（補充：Reviewer 模型決策、Coder 自我確認偏誤）  
> 範圍：設計方向確認，不含實作 code  
> 適用分支：在現有 LangGraph StateGraph（agent.py）+ Chainlit（app.py）架構上疊加

---

## 前置：現有架構摘要

閱讀程式碼後確認的現況：

| 元件 | 實作位置 | 行為 |
|------|----------|------|
| Coder Agent（大腦） | `agent.py / call_model` | `gemma4:26b` via ChatOllama，bind_tools 全部工具 |
| 工具節點 | `agent.py / tool_node` | LangGraph ToolNode，執行 safe + dangerous 工具 |
| 危險工具攔截 | `agent.py / pre_tool_check` + `interrupt_before` | no-op 節點，upload_to_youtube / delete_local_video 在此暫停 |
| 人工審查 UI | `app.py / _handle_interrupt` | Chainlit AskActionMessage，approve/reject |
| 程式碼安全工具 | `agent.py / forge_and_test_tool` | AST 掃描 → 沙盒 unittest → 寫入 custom_tools.py → reload |
| 圖狀態 | `MessagesState` | 純 messages 陣列，無額外欄位 |

現有強制規則（SYSTEM_PROMPT）已有「禁止用純文字假裝執行工具」的方向，
本設計在此精神上延伸，不另起爐灶。

---

## 1. 結構化任務回報 Schema

### 1.1 決策：新增強制性 `submit_task_completion` 工具

**選定方案：** 定義一個新 tool — `submit_task_completion`，大腦完成任務後
**必須呼叫此工具**才算「提交」，否則不進入下一流程。

欄位設計（以 Pydantic 欄位描述，此階段不寫程式碼）：

```
TaskCompletionReport
│
├── task_id          : str         # UUID，由呼叫時自動生成，用於對應後續審查記錄
├── task_description : str         # 大腦對「本次任務目標」的一句話摘要（複述使用者需求）
├── modified_files   : list[str]   # 確實被改動的檔案路徑清單；若無改動則為空清單
├── change_summary   : str         # diff 的精簡描述（100字以內），不貼全文
├── test_executed    : bool        # 是否呼叫過 forge_and_test_tool 或其他測試工具
├── test_passed      : bool | None # True/False；若 test_executed=False 則填 None
├── risk_level       : "low" | "medium" | "high"
│                                  # 大腦自評：見下方評分標準
├── risk_reason      : str         # 自評理由（一句話說明為何是這個等級）
└── tool_calls_made  : list[str]   # 本次任務實際呼叫過的工具名稱清單
```

**risk_level 自評標準（寫入 system prompt）：**

| 等級 | 條件 |
|------|------|
| `low` | 只有讀操作或查詢；無任何檔案寫入或外部 API 呼叫 |
| `medium` | 有寫入 custom_tools.py 或資料庫，但有單元測試通過 |
| `high` | 任何以下情況：測試未通過、改動涉及安全邏輯、呼叫危險工具、修改現有工具定義 |

---

**理由：** 這與現有「禁止純文字假裝執行工具」的精神完全一致。
現有的 `forge_and_test_tool` 已證明「以 tool call 作為強制 gate」的模式可行。
`submit_task_completion` 是同一套機制的延伸：
把「任務完成」本身也做成一個必須呼叫的 tool，而非讓大腦自由發揮結尾文字。

---

**拒絕的替代方案：**

| 方案 | 拒絕理由 |
|------|----------|
| 後處理解析（parse 最後一則 assistant message） | 大腦輸出格式不穩定，尤其 gemma4:26b 在繁中場景下格式更難預測；這正是要解決的問題本身 |
| 任務完成後追加獨立 prompt 要大腦「再輸出一次結構化版本」 | 同一 context 延續，大腦有動機美化或修正已輸出的內容，無法防止事後竄改 |
| 在 Chainlit 層用 regex 解析 | 繞過了 LangGraph 的狀態機，後續節點無法從 state 取得結構化資料 |

---

### 1.2 Schema 解析失敗的處理

**決策：** 三段式降級處理

```
嘗試解析 submit_task_completion 的參數
│
├── 成功 → 正常進入 Reviewer
│
├── 第一次失敗（schema validation error）
│     → 注入系統訊息「上次提交格式錯誤，原因：{error}，請重新呼叫 submit_task_completion」
│     → 給大腦第二次機會重新呼叫
│
└── 第二次仍失敗
      → 將任務標記為 parse_error
      → 強制觸發 interrupt_before（與 high-risk 路徑相同）
      → 在 Chainlit 顯示警示：「大腦無法產生結構化回報，需要人工確認」
```

**理由：** 給一次重試機會可處理輸入格式小失誤，但兩次失敗代表系統性問題，
應當人工接手，不能無限重試。

---

## 2. 獨立 Reviewer Agent 的隔離設計

### 2.1 決策：完全隔離的第二次 LLM 呼叫

**選定方案：** Reviewer Agent 是一個新的 LangGraph 節點（`reviewer_agent`），
在其中發出**全新的** `llm.ainvoke`（或 `llm.astream`）呼叫，
**不傳入任何來自 Coder Agent 對話的 messages history**。

Reviewer 收到的最小資訊包：

```
ReviewerInput
│
├── original_task     : str   # 使用者原始指令的完整文字（從對話第一則 user message 取得）
├── change_summary    : str   # 來自 TaskCompletionReport.change_summary
├── modified_files    : list[str]
├── test_executed     : bool
├── test_passed       : bool | None
├── risk_level_self   : str   # 大腦自評等級（供 Reviewer 交叉驗證，而非盲目接受）
├── ast_scan_result   : str | None
│     # 若本次任務有呼叫 forge_and_test_tool，
│     # 把該工具的回傳字串（含 AST 掃描結果與測試輸出）帶給 Reviewer
└── tool_calls_made   : list[str]
```

**刻意排除的資訊：**
- Coder Agent 的推理過程（任何 assistant 訊息）
- Coder Agent 對自己工作品質的主觀評價
- 中間步驟的 tool response 內容（除 forge_and_test_tool 外）

**理由：** 確認偏誤（confirmation bias）的核心機制是：
審查者看到了被審查者的推理脈絡後，傾向於為這個脈絡找支持點。
只給「原始需求 + 結果」讓 Reviewer 獨立判斷是否吻合，才能打破這個偏誤。

---

### 2.2 Reviewer Agent 的 System Prompt 人設設計

**選定方案（人設核心）：**

```
你是一位嚴格的程式碼審查員（Code Reviewer），職責是找出問題，不是讚美。
你的唯一目標是保護系統不被引入缺陷、安全漏洞、或未完成的實作。

行為規範：
1. 凡有疑慮，必須列為 issue，不得因「看起來應該 OK」而放行。
2. 你看不到實作者的推理，這是刻意設計的——你的判斷必須完全基於改動的結果，
   而非實作者的意圖說明。
3. 若改動明顯超出原始任務範圍（scope creep），視為 blocker。
4. 若有任何欄位資訊不足以判斷，在 issues 中明確標記「資訊不足，需要人工確認」。
5. 最後輸出必須嚴格遵守指定的 JSON schema，不得附加自由文字。
```

**理由：** 預設友善的審查員在 LLM 場景下幾乎必然輸出「基本上沒問題，但有幾個小建議」，
這在 Coder 與 Reviewer 共用同一底層模型時尤其嚴重（同一個 gemma4:26b，
風格與偏好相同）。明確指定「職責是找問題」能校正這個系統性偏差。

---

### 2.3 Reviewer 應檢查的具體維度

| 維度 | 具體判斷依據 |
|------|------------|
| **任務符合性** | `original_task` vs `change_summary` — 改動是否真的解決了使用者的原始需求？是否有遺漏？是否有超出範圍的改動（scope creep）？ |
| **安全風險** | `tool_calls_made` + `ast_scan_result` — 若呼叫了 `forge_and_test_tool`，AST 掃描是否通過？`change_summary` 中是否有潛在的注入風險或不受控的外部呼叫？ |
| **邊界案例** | 依 `change_summary` 判斷：空輸入、檔案不存在、網路逾時、JSON 解析失敗等邊界是否有明確處理或測試覆蓋 |
| **測試狀態** | `test_executed + test_passed` — 若 test_executed=False 且 modified_files 不為空，視為自動加一個 issue（severity: major）；若 test_passed=False，自動加 blocker |
| **自評 risk_level 合理性** | Reviewer 需獨立評估 risk_level 是否合理，若與大腦自評不一致須標記 |

---

### 2.4 Reviewer 輸出 Schema

```
ReviewResult
│
├── verdict          : "approve" | "needs_changes"
├── reviewer_risk    : "low" | "medium" | "high"   # Reviewer 自行評估的風險等級
├── issues           : list[Issue]                  # 空清單表示無問題
├── approved_aspects : list[str]                    # 明確確認 OK 的面向（≥1 條）
└── one_line_summary : str                          # 給使用者看的一句話結論

Issue
│
├── severity    : "blocker" | "major" | "minor"
├── category    : "correctness" | "security" | "edge_case" | "test" | "scope"
├── description : str         # 問題描述
├── location    : str | None  # 例如 "custom_tools.py" 或 "send_email tool"
└── suggestion  : str | None  # 建議修正方向（非強制）
```

**verdict 判定規則（在 Reviewer system prompt 中明確說明）：**
- 任何 `blocker` issue → 必須 `needs_changes`
- 有 `major` issue 且數量 ≥ 2 → `needs_changes`
- 只有 `minor` issue → 可以 `approve`，但 issues 仍然列出

---

### 2.5 Reviewer Agent 使用的模型（明確決策）

**決策：使用 `qwen2.5-coder:7b`（Ollama 本地模型），不使用同一個 gemma4:26b。**

> **選型歷程記錄（2025-06 實測後確定）：**
>
> | 模型 | 結果 | 淘汰原因 |
> |------|------|----------|
> | `starcoder2:7b` | ❌ JSON 合法性 0/3 | 純 code completion 模型，無 instruction following，每次生成不相關的 Markdown 內容，100% 觸發 fallback |
> | `codellama:7b-instruct` | ❌ 審查品質 0/3 | 能輸出合法 JSON（3/3），但推理能力不足，三個測試案例（邏輯錯誤、指令注入、答非所問）全部誤判 approve，回應只是複述 change_summary |
> | `qwen2.5-coder:7b` | ✅ **最終選定** | Instruction-tuned + 程式碼專精 + 推理能力足夠 + 7B 輕量 + 訓練血統與 gemma 獨立 |
>
> **Reviewer 必須是 instruction-tuned 模型**——這是當初選型時漏掉的必要前提。
> 純 code completion 模型（starcoder2）無法遵守「只輸出 JSON」的格式要求。
>
> **資安考量確認（2025-06）：** 地端 Ollama 離線推理，資料不外傳；
> 本機 Ollama 已確認綁定 localhost，未對外暴露。
> 模型來源的資安疑慮在全本地架構下不成立，qwen2.5-coder:7b 採用。

---

**為什麼 Option A（同模型 + 隔離 context）不夠？**

v0.1 的 2.1 節只解決了「Reviewer 看到了 Coder 的推理所以被帶著走」這一種確認偏誤，
但還有第二種：**訓練偏誤（calibration bias）**。

同一個 gemma4:26b 有同樣的訓練語料、同樣的評估標準、同樣的盲點。
若 gemma4:26b 在訓練中系統性地低估了某類問題（例如：對不處理 None 的函式
傾向於認為「OK，呼叫者會負責」），那麼 Coder 和 Reviewer 會共享這個盲點，
不論 context 怎麼隔離。

Context 隔離解決的是「我知道自己寫了這段，所以我護著它」，
解決不了的是「我訓練出來就不認為這是問題」。

估計效益：context 隔離大約能消除 60–70% 的確認偏誤風險；
換用不同模型則接近 90%（兩者互補的盲點不同，更容易被對方抓出來）。

---

**Reviewer 模型必備條件（選型前提）：**

1. **Instruction following 能力**：必須能讀懂系統提示並嚴格遵守輸出格式，
   尤其是「只輸出 JSON，不得附加說明」。純 code completion 模型（如 starcoder2）
   不具備此能力，無法作為 Reviewer。
2. **訓練血統與 Coder（gemma4:26b）不同**：目的是互補盲點，同源模型審查同源輸出無意義。
3. **程式碼理解能力**：需理解程式碼結構、安全問題、邏輯正確性，通用語言模型適用，
   程式碼專精模型更佳。
4. **全本地執行**：不依賴外部 API，符合現有 Ollama 架構。

---

**為什麼選 `qwen2.5-coder:7b`？**

| 考量 | 說明 |
|------|------|
| **Instruction tuned** | 經過 RLHF 微調，能讀懂並嚴格遵守「只輸出 JSON」等格式要求——這是 starcoder2 致命缺乏的能力 |
| **程式碼推理能力足夠** | Qwen2.5-Coder 系列在程式碼理解與 bug 偵測基準上表現優異，能比對「原始任務 vs 實際 diff」並識別邏輯錯誤、安全漏洞、scope 不符——這是 codellama:7b-instruct 實測確認不足的能力 |
| **訓練血統與 gemma 獨立** | Qwen 系列由阿里雲訓練，與 Google gemma4 的訓練語料及 RLHF 偏好完全不重疊，系統性盲點互補 |
| **7B 輕量** | 約 4.5 GB VRAM；gemma4:26b 約 15 GB VRAM；序列執行，峰值 VRAM 是 max(15, 4.5) = 15 GB |
| **資安確認無虞** | 地端 Ollama 離線推理，程式碼不外傳；本機已確認綁定 localhost，未對外暴露 |
| **保持全本地** | 現有架構以 Ollama 為核心，不依賴外部 API |

---

**拒絕的替代方案：**

| 方案 | 拒絕理由 |
|------|----------|
| **Option A：同 gemma4:26b + 隔離 context** | 只能消除 context 層級的確認偏誤，無法消除訓練層級的系統性盲點 |
| **starcoder2:7b** | 純 code completion 模型，無 instruction following，JSON 合法性 0/3（實測確認） |
| **codellama:7b-instruct** | JSON 合法性 3/3，但審查品質 0/3，推理不足全部誤判 approve（實測確認） |
| **Claude API** | 程式碼需外傳，現有架構設計為全本地；有 API 費用；離線場景下不可用 |
| **deepseek-coder 系列** | 由深度求索（DeepSeek）開發，排除此訓練來源 |

---

**資源負擔總結（供實作參考）：**

```
主機 VRAM    │ 行為
─────────────┼─────────────────────────────────────────────────────
< 16 GB      │ gemma4:26b 需 CPU offload；加 7B reviewer 影響有限
16–20 GB     │ gemma4:26b 常駐；reviewer 每次需 10–30 秒切換載入
≥ 20 GB      │ 兩個模型可同時常駐；切換 < 2 秒
```

---

## 3. 修正迴圈設計

### 3.1 LangGraph 狀態轉移設計

**新增的 State 欄位（擴充現有 MessagesState）：**

```
DaedalusState（繼承 MessagesState）
│
├── task_report    : TaskCompletionReport | None
├── review_result  : ReviewResult | None
├── retry_count    : int   # 預設 0，每次 needs_changes 後 +1
└── original_task  : str   # 對話中第一則 user message 的內容（初始化時快照）
```

**狀態轉移圖：**

```
START
  │
  ▼
[agent] ◄────────────────────────────────────────────────────────┐
  │                                                               │
  ├─ tool_calls: dangerous → [pre_tool_check] → [tools] ─────────┘
  ├─ tool_calls: safe      → [tools] ─────────────────────────────┘
  ├─ tool_call: submit_task_completion
  │     │
  │     ▼
  │  [task_reporter]  ← 解析 schema / 失敗處理
  │     │
  │     ├─ parse_error × 2 → [human_escalation]
  │     │
  │     └─ 解析成功
  │           │
  │           ▼
  │        [reviewer_agent]  ← 獨立 LLM 呼叫
  │           │
  │           ├─ verdict=approve + risk=high
  │           │     └─→ [human_escalation]（強制人工確認）
  │           │
  │           ├─ verdict=approve + risk=low/medium
  │           │     └─→ END
  │           │
  │           ├─ verdict=needs_changes + retry_count < N
  │           │     └─→ [agent]（注入 ReviewResult 作為新 user message）
  │           │
  │           └─ verdict=needs_changes + retry_count ≥ N
  │                 └─→ [human_escalation]
  │
  └─ no tool_calls → END（閒聊或查詢，不走 reporter 流程）


[human_escalation]
  ├── 在 Chainlit 顯示完整審查記錄
  ├── 觸發 interrupt_before 等效機制（AskActionMessage）
  └── 使用者選擇：繼續 / 放棄 / 手動修正後重試
```

**重試時的 Context 重建策略（「選擇性失憶」）：**

> 詳細設計理由見 3.4 節。這裡只列操作規格。

重試時，**不**直接把 ReviewResult 追加到現有 messages history 末尾。
而是在 `reviewer_agent` 路由到 `agent` 之前，執行一次 **context surgery**：

```
重建後的 messages 序列
│
├── [user]   original_task（原始使用者指令，保留）
├── [tool]   第一次嘗試所有 ToolMessage 結果（保留）← 客觀事實
│              例：forge_and_test_tool 回傳、add_japanese_word 回傳
├── [user]   Reviewer 審查結果（注入，作為新的指令）
│
└── 刪除全部 [assistant] 推理訊息（Coder 自己的分析文字全部移除）
```

注入的 Reviewer 訊息格式：

```
[Reviewer 審查結果 — 第 {retry_count} 次 / 共 3 次]

結論：{one_line_summary}

問題清單：
- [blocker] {issue.description}（{issue.location}）
  建議：{issue.suggestion}
- [major] ...

以上工具執行記錄（tool messages）已保留供參考。
請重新審視原始任務，修正後再次呼叫 submit_task_completion 提交。
```

---

### 3.2 迴圈上限 N 的選擇

**決策：N = 3**

| N 值 | 評估 |
|------|------|
| N=1 | 太嚴苛，小錯誤一次就升級人工，無謂打擾 |
| N=2 | 仍偏嚴，Reviewer 可能在第一次遺漏了某些需要大腦先修正才能再次評估的問題 |
| **N=3** | 三次給大腦充分機會，同時避免無限迴圈消耗資源（gemma4:26b 是本地模型，CPU/GPU 有限） |
| N=5+ | 若三次都過不了審查，多半是根本的理解偏差，繼續重試不會改善 |

N=3 次全部失敗後的處理：

1. 將任務狀態標記為 `escalated`
2. 彙整所有三次的 TaskCompletionReport + ReviewResult 形成完整審查歷程
3. 觸發 `human_escalation`（同 interrupt_before 機制）
4. 在 Chainlit 顯示完整三次嘗試的對比摘要，讓使用者一眼看清問題卡在哪裡

---

### 3.4 Coder Agent 重試時的自我確認偏誤問題

**問題描述：**

v0.1 的 2.1 節花了完整篇幅說明為何 Reviewer 必須隔離於 Coder 的對話歷史之外，
理由是「看過 Coder 推理的 Reviewer 會被帶著走，產生確認偏誤」。

但 v0.1 的重試設計卻讓 Coder 在第二次嘗試時，仍保有自己第一次的
**完整推理脈絡**（所有 assistant messages），這會造成鏡像問題：
**Coder 看著自己第一次的推理，傾向於只做表面修正，而非真正重新思考。**

---

**這個偏誤確實存在嗎？**

是，而且在 LLM 場景下尤其明顯。模式如下：

> 「我上次選擇方案 X，是因為理由 A、B、C，這些理由我依然認為成立，
>  所以 Reviewer 的問題應該只是技術細節——加一個 try/except 就好了。」

這是 LLM 版本的「固執於既有認知框架（anchoring）」。
Reviewer 的具體 issue 被當成需要「修補」的點，而非重新考慮整個方法的訊號。

---

**v0.1 沒有處理這個問題，原因分析：**

v0.1 設計重試時，把問題想成「Coder 需要知道哪裡錯了才能修正」，
所以注入 ReviewResult 就夠了。這個假設在錯誤是「技術細節」時成立，
但在根本方向有誤時失效——因為 Coder 的舊推理會優先於 Reviewer 的批評。

---

**決策：重試採用「選擇性失憶（Selective Amnesia）」**

**選定方案：** 重試時重建 Coder 的 context，
**移除所有 assistant 推理訊息，保留 tool call 結果（ToolMessages）**。

保留 vs 移除的分界線：

| 訊息類型 | 動作 | 理由 |
|----------|------|------|
| `user` — 原始任務 | **保留** | 這是 Coder 需要重新對焦的目標 |
| `tool` — 工具執行結果 | **保留** | 這是客觀事實（AST 分析通過、DB 寫入成功等），Coder 需要知道當前系統狀態 |
| `assistant` — 推理文字 | **移除** | 這是 Coder 的判斷與詮釋，正是造成固執的來源 |
| `user/system` — Reviewer 結果 | **注入** | 作為重試的新起點指令 |

---

**非冪等工具的重複執行防護**

選擇性失憶移除了 assistant 推理訊息，但 Coder 仍看得到 ToolMessages（工具執行結果）。
問題在於：**工具名稱不在 ToolMessage 本身，而在被刪除的 AIMessage.tool_calls 裡**。
移除 AIMessages 後，Coder 看到的 ToolMessage 內容是孤立的字串，
無法自行判斷「這個結果屬於哪個工具」。

更嚴重的是，現有工具中有兩個是**非冪等**的（重複呼叫會產生額外副作用）：

| 工具 | 非冪等行為 | 重複呼叫的後果 |
|------|-----------|--------------|
| `forge_and_test_tool` | append 寫入 `custom_tools.py` | 同一個函式被寫入兩次，動態 reload 時名稱衝突 |
| `generate_japanese_learning_video` | 建立新 video_batch DB 記錄 + 落地 MP4 | 重複生成同一批次的影片，浪費大量 GPU 時間與磁碟空間 |

解法：**context surgery 分兩步驟，第一步在刪除 AIMessages 之前提取工具執行摘要。**

---

*步驟一：提取（在刪除 AIMessages 之前執行）*

```
1. 掃描全部 AIMessages
   → 建立對照表：{ tool_call_id → tool_name }
     （來源：AIMessage.tool_calls 的每個 entry 含 id 和 name）

2. 掃描全部 ToolMessages
   → 對每筆 ToolMessage：
       name    = 對照表[tool_call_id]
       success = content 中不含以下任一詞：
                 "失敗"、"錯誤"、"Error"、"error"、"Exception"、"無法"
       → 記錄為 (name, success, content 前 80 字)

3. 分類
   NON_IDEMPOTENT = {"forge_and_test_tool", "generate_japanese_learning_video"}
   → 依 name 是否在 NON_IDEMPOTENT 集合中分為「有副作用」vs「安全重複」
```

*步驟二：注入摘要（連同 Reviewer 結果一起注入）*

格式設計（純文字，附加在 Reviewer 結果訊息的前段）：

```
[已完成操作摘要 — 重試請確認後再執行]

★ 有持久副作用（非冪等，請勿重複呼叫）：
  - forge_and_test_tool × 1  ✅ 成功
    已 append 寫入 custom_tools.py 並動態載入。
    若需修改工具邏輯，必須先確認不重複定義相同函式名稱。

◎ 安全重複（冪等或純讀取）：
  - add_japanese_word × 5   ✅ 全部成功（upsert，重複無害）
  - web_search × 2          ✅ 已執行

✗ 未成功 / 未執行（本次重試的目標範圍）：
  - （依 Reviewer 問題清單決定）

---
[Reviewer 審查結果 — 第 {retry_count} 次 / 共 3 次]
...
```

格式設計原則：
- 非冪等工具**總是列出**，無論成功或失敗，並附帶一行警語說明副作用。
- 冪等工具只在成功時列出（讓 Coder 知道「這些已完成，不需要重跑」）；
  失敗的冪等工具不列出，讓 Coder 自然地在重試中重新嘗試。
- 格式刻意用純文字而非 JSON，因為這段摘要的閱讀對象是 Coder，
  不需要機器解析，可讀性優先。

---

**這個設計與 2.1 節是否一致？**

是，且是同一個原則在兩個不同方向的應用：

```
2.1 節的問題：Reviewer 看到 Coder 的推理 → Reviewer 被帶著走
              解法：不給 Reviewer 看 Coder 的推理

3.4 節的問題：Coder 看著自己的推理 → Coder 固執於既有方向
              解法：重試時移除 Coder 自己的推理，從 tool results 重新出發
```

兩者都是「先前的推理脈絡會錨定（anchor）後續的判斷」，
解法都是「斷開這個錨」。

v0.1 在 Reviewer 端做了，卻在 Coder 端遺漏了。
v0.2 在兩端都套用相同機制，設計更一致。

---

**拒絕的替代方案：**

| 方案 | 拒絕理由 |
|------|----------|
| **保留全部歷史（v0.1 原設計）** | Coder 有完整的自我推理在眼前，大概率只做最小化修正（表面補丁），通不過根本問題的修正 |
| **完全清空，只傳 original_task + ReviewResult** | 過於激進：Coder 不知道自己上次呼叫了哪些工具、工具回傳什麼狀態，可能重複呼叫已成功的工具（浪費資源），或不知道系統當前實際狀態 |
| **保留全部歷史 + 在 system prompt 中特別強調「不要守舊」** | Prompt 指令對 LLM 的錨定效應影響有限，尤其是在推理文字已經在 context 中的情況下 |

---

**實作注意：** 在 `reviewer_agent` 路由至 `agent` 的 edge 上，
需要對 `DaedalusState.messages` 做一次過濾操作（移除 role == "assistant" 的訊息），
再附加 Reviewer 結果訊息，然後才調用 `call_model`。
這是一個 context surgery 步驟，不在本次設計範圍，但需標記為實作時的要點。

---

### 3.3 risk_level=high 的強制人工確認

**決策：是，無論 Reviewer verdict 如何，risk_level=high 一律觸發 human_escalation。**

理由：

- `high` 的定義已包含「測試未通過、改動涉及安全邏輯、呼叫危險工具」—
  這些情況即使 Reviewer 認為 OK，其判斷本身也需要人類驗證。
- Reviewer Agent 本身也是 LLM，在高風險場景下引入一個 LLM 審核另一個 LLM，
  不能完全替代人類判斷。
- 與現有 `interrupt_before` 設計精神一致：危險操作必須人類知情。

**拒絕的替代方案：** 讓 Reviewer 自主決定 high-risk 任務是否需要人工 —
這把過多的責任交給了 LLM，與整個機制設計的前提矛盾。

---

## 4. 與現有 Chainlit 介面的整合

### 4.1 任務完成摘要卡（Task Summary Card）

**決策：** 使用 Chainlit 的 `cl.Message` + Markdown 渲染，
以結構化的卡片格式呈現 TaskCompletionReport，**不直接顯示 raw JSON**。

卡片格式設計：

```markdown
## 任務回報

**目標**：{task_description}

**改動檔案**
- custom_tools.py
- encyclopedia.py

**測試狀態**：✅ 通過（呼叫了 forge_and_test_tool）

**風險等級**：🟡 medium — 有寫入 custom_tools.py，但測試通過

**使用工具**：forge_and_test_tool → submit_task_completion
```

顏色/圖示規則：
| 風險等級 | 圖示 |
|----------|------|
| low | 🟢 |
| medium | 🟡 |
| high | 🔴 |

---

### 4.2 Reviewer 審查結果的視覺化呈現

**approve（無 blocker/major issues）：**

```markdown
## ✅ Reviewer 審查通過

風險等級（Reviewer 評估）：🟢 low

確認 OK 的面向：
- 改動確實解決了原始需求
- 邊界案例有 try/except 處理

次要建議（不影響核准）：
- [minor] 可考慮加入日誌記錄
```

**needs_changes：**

```markdown
## ⚠️ Reviewer 要求修正（第 1 次 / 共 3 次）

結論：改動未完整解決原始任務要求

問題清單：
🔴 [blocker] correctness — 空輸入時 word_json 解析未保護邊界
   位置：add_japanese_word tool
   建議：在 json.loads 外加 try/except 並回傳明確錯誤訊息

🟠 [major] edge_case — 網路逾時場景未處理
   位置：fetch_web_page tool
```

**human_escalation 畫面（三次失敗後）：**

Chainlit 顯示三次嘗試的摺疊式對比（`cl.Accordion` 等效），
加上一個明確的 `AskActionMessage`：

- 選項 A：指示大腦重新理解任務（清空 retry_count，重新開始）
- 選項 B：使用者手動說明如何修正
- 選項 C：放棄此次任務

---

## 5. 與現有安全機制的關係

### 5.1 兩套機制的定位

**決策：** 兩套機制是互補的，不替代彼此，各自守住不同的 gate。

| 機制 | 守住的 Gate | 觸發時機 | 性質 |
|------|-------------|----------|------|
| `interrupt_before + AskActionMessage` | **危險副作用 Gate** | 大腦即將呼叫 upload_to_youtube / delete_local_video **之前** | 防止不可逆的外部操作 |
| Reviewer Agent + human_escalation | **品質 + 安全 Gate** | 大腦呼叫 submit_task_completion **完成任務之後** | 防止有缺陷的程式碼進入系統 |

兩套機制不重疊，也不替代：
- Reviewer 審查通過 ≠ 危險工具可以自動執行
- 即使 Reviewer approve，後續的 upload_to_youtube 依然會觸發 interrupt_before

---

### 5.2 執行順序說明

以「大腦撰寫新工具 → 上傳影片」完整流程為例：

```
1. 使用者下達指令
2. 大腦執行 forge_and_test_tool（AST + 沙盒 + 動態載入）
3. 大腦呼叫 submit_task_completion
       │
       ▼
4. [task_reporter] 解析 schema
5. [reviewer_agent] 獨立審查（不知道大腦的推理）
6a. needs_changes → 回到大腦修正（最多 3 次）
6b. approve + risk=medium → 繼續流程
7. 大腦呼叫 generate_japanese_learning_video
8. 大腦呼叫 upload_to_youtube
       │
       ▼（interrupt_before 觸發）
9. Chainlit AskActionMessage：「即將執行危險操作，是否核准？」
10. 使用者核准 → 執行上傳
```

Reviewer 審查的是「步驟 2-3 的工作品質」，
interrupt_before 攔截的是「步驟 8-10 的不可逆副作用」。
兩者各司其職。

---

### 5.3 submit_task_completion 本身是否需要 interrupt_before？

**決策：不需要。**

`submit_task_completion` 本身是純輸出（報告），不執行任何副作用，
不寫檔案、不呼叫外部 API。它觸發的是 Reviewer Agent 流程，
Reviewer 也只讀取資料、不執行操作。這整段流程是「觀察者模式」，
不需要額外的人工攔截點（人工審查已在 human_escalation 環節補上）。

---

## 附錄：待定問題（留給下一次討論）

1. **`original_task` 的定義範圍？** 若使用者在多輪對話後才下達最終任務指令，
   `original_task` 應取哪一則 user message？需定義「任務開始」的錨點。
   候選方案：(a) 永遠取最新一則 user message；(b) 在 submit_task_completion 的
   schema 中讓 Coder 自行填寫 task_description，以此作為錨點；
   (c) 透過 DaedalusState 快照第一則未被前次任務覆蓋的 user message。

2. **Reviewer 的 `ast_scan_result` 資訊格式？** `forge_and_test_tool` 的回傳
   是純字串，需要定義如何切割「AST 結果部分」傳給 Reviewer，
   而不是把整個 unittest 輸出都帶進去。
   候選方案：在 forge_and_test_tool 回傳格式中明確標記 `[AST_RESULT]` 分段，
   讓 task_reporter 節點用標記切割後只取 AST 部分。

3. **`DaedalusState` 的擴充是否影響現有 `MemorySaver` checkpointing？**
   需要確認新增欄位後 state schema 的序列化相容性。LangGraph 的 MemorySaver
   使用 Python pickle，新增 Pydantic model 欄位通常相容，
   但需要實際測試 None 預設值的 checkpoint 讀回行為。

