# Roadmap

整體目標：建立一個以 **Laya 當 policy / value 網路**、用 **MCTS 做搜尋**的 AlphaZero / Expert Iteration 式自我改進系統，先在可驗證的玩具問題上證明迴圈有效，再推進到文字代理任務與雙人對抗問題。候選問題的分析與選擇理由見 [`candidate_problems.md`](candidate_problems.md)，目前的執行進度見 [`TODO.md`](TODO.md)。

狀態圖例：✅ 完成　🟡 進行中　⏸ 受阻　⬜ 未開始

---

## Phase 0：基礎建設與管線驗證　🟡

**目標**：打通「teacher 冷啟動 → MCTS self-play → 軟目標 → Laya 微調 → 校準 → 評估」整個迴圈，並在兩個便宜、可完美驗證的問題上確認它真的會讓模型進步。

| 項目 | 狀態 |
|---|---|
| 環境：Python 3.11 venv、`laya==0.3.24`、torch、SymPy、測試工具；`scripts/setup_env.sh`、`mcts-laya download` | ✅ |
| 模組化套件：envs / evaluators / search / selfplay / teachers / training / pipeline，以 registry 解耦 | ✅ |
| 任務 B：Countdown 環境 + 窮舉 teacher | ✅ |
| 任務 A：一元一次方程逐步改寫環境（SymPy 驗證）+ BFS teacher | ✅ |
| Laya 批次評估器（繞過 `predict_batch` 同題限制） | ✅ |
| PUCT（virtual loss 批次）與 Gumbel AlphaZero 搜尋 | ✅ |
| 訓練器：soft CE + RLCD、選項順序擴增、溫度校準、輸出 Laya checkpoint | ✅ |
| 微型 Laya 相容模型（離線 / CPU 開發與測試） | ✅ |
| 微型模型上跑完兩個任務的 Phase 0 實驗 | ✅ |
| **以真正的 Laya 權重跑 Phase 0** | ⏸ 容器無法連到 huggingface.co |

**里程碑（每個任務都要通過）**
1. Laya+MCTS 的評估成功率 / reward 隨迭代上升
2. Laya+MCTS 勝過 Laya greedy（不搜尋）
3. Laya+MCTS 勝過同預算、均勻先驗的 MCTS（不用 Laya）

---

## Phase 1：文字代理主線任務　⬜（6–8 週）

**目標**：在真正的文字環境上，證明「小型 System 1 模型 + 搜尋」能在成本與延遲上勝過 LLM agent。

- 環境擇一：**ALFWorld**（有 expert 軌跡）或 **WebShop**（已有 Laya 瀏覽器微調的前例）；TextWorld / Jericho 作為備選
- 有狀態模擬器的 adapter（save/restore 或動作重播）、歷史摘要（context 512–1024）
- 動作數 > 20 時的 shortlist / 階層式 choice / progressive widening
- 多 episode 並行，跨 episode 合併葉節點批次（GPU 吞吐量）；搜尋樹重用；ONNX 推論
- MuZero Reanalyse、凍結 encoder / LoRA 等降低迭代成本的手段
- 對照組：ReAct / LATS（LLM 當 policy / value），比較成功率、每任務成本、延遲

**出口條件**：在選定環境上，Laya+MCTS 達到可報告的成功率，並畫出相對 LLM 基準的成本–成功率 Pareto 前緣。

## Phase 2：雙人對抗 / 領域應用　⬜（8–12 週）

- **CybORG / CyberBattleSim 紅藍隊自我對弈**：狀態文字化、league training、Elo 曲線（搜尋核心已經支援雙人零和）
- 或 **多跳檢索問答**（HotpotQA / MuSiQue），並發展中文版本（`laya-multilingual`）

**出口條件**：真正的 AlphaZero 式對抗學習曲線（Elo 隨迭代上升），或領域任務上相對基準的顯著提升。

## Phase 3（長期）：LLM 提案 + Laya 價值　⬜

- Lean 4 定理證明（LeanDojo / miniF2F）：tactic 由 LLM 提出，Laya 負責重排序與價值估計
- 一般化的「System 2 提案 / System 1 評估」框架（新的 `Evaluator` 加上提案器）
