# Roadmap

整體目標：建立一個以 **Laya 當 policy / value 網路**、用 **MCTS 做搜尋**的 AlphaZero / Expert Iteration 式自我改進系統，先在可驗證的玩具問題上證明迴圈有效，再推進到文字代理任務與雙人對抗問題。候選問題的分析與選擇理由見 [`candidate_problems.md`](candidate_problems.md)，目前的執行進度見 [`TODO.md`](TODO.md)。

狀態圖例：✅ 完成　🟡 進行中　⏸ 受阻　⬜ 未開始

---

## Phase 0：基礎建設與管線驗證　✅（報告：[`docs/phase0_report.md`](docs/phase0_report.md)）

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
| 以真正的 Laya 權重（multilingual）跑 Phase 0：CPU 縮小版 | ✅ |
| 完整規模（評估 100 題、self-play 每輪 64 局）的真權重實驗 | ⬜ 需要 GPU |

**里程碑（每個任務都要通過）**
1. Laya+MCTS 的評估成功率 / reward 隨迭代上升
2. Laya+MCTS 勝過 Laya greedy（不搜尋）
3. Laya+MCTS 勝過同預算、均勻先驗的 MCTS（不用 Laya）

**結果**：Countdown（真 Laya）三項全過，0.93 vs 均勻 0.30；代數（微型模型，弱 warm start 與從零開始）三項全過；代數（真 Laya）warm start 後即達最優解。真 Laya 從零開始的代數 4 輪內只過第 1 項，需要更多迭代。

---

## Phase 1：文字代理任務　🟡

**目標**：在真正的文字環境上，證明「小型 System 1 模型 + 搜尋」能學會需要探索與記憶的多步任務，並在成本與延遲上和 LLM agent 比較。分成兩段：先在 TextWorld 上用可控難度的課程建立能力（1a），再推進到標準 benchmark ALFWorld（1b）。兩者使用同一個引擎，adapter 可以共用。

### Phase 1a：TextWorld（主線）　🟡

| 項目 | 狀態 |
|---|---|
| `Environment` 介面支援資料切分（`split`：train / eval），評估一律用保留的遊戲 | ⬜ |
| TextWorld adapter：函數式狀態（遊戲 + 動作歷史）、reset + 前綴重播（不用 52 ms 的 `copy()`）、指令過濾 | ⬜ |
| 狀態文字表示：目標、最近動作、物品欄、所在位置、上一步結果、剩餘步數 | ⬜ |
| 關卡與遊戲池：`mcts-laya tw-games` 平行產生 train / eval 遊戲（不同 seed） | ⬜ |
| teacher：使用 TextWorld 的 `policy_commands`（每個狀態的最佳剩餘指令） | ⬜ |
| 實驗：L1 → L2 → L2-goal → L3-goal（真 Laya，CPU 縮小版） | ⬜ |
| cooking 關卡（FTWP 題型）作為通往 ALFWorld 的橋接 | ⬜ |

關卡（房間 / 物件 / 任務步數，目標寫法）：L1 = 3/6/3 完整指示；L2 = 5/10/5 完整指示；L2-goal = 5/10/5 只給最終目標；L3-goal = 8/15/8 只給最終目標。

**出口條件**：在 L2-goal 與 L3-goal 上，Laya+MCTS 勝過 Laya greedy 與同預算的均勻先驗 MCTS，且 self-play 讓成功率上升。

### Phase 1b：ALFWorld（進階目標）　⬜

- 沿用 1a 的 adapter，加上動作數 > 20 時的 shortlist / 「動詞 → 目標」兩層 choice
- 用 ALFWorld 的 PDDL expert 冷啟動
- 和 ReAct / Reflexion（LLM 當 policy）比較成功率、每任務成本、延遲，畫出 Pareto 前緣
- 效能：多 episode 並行、跨 episode 合併葉節點批次、搜尋樹重用、MuZero Reanalyse、凍結 encoder / LoRA

**出口條件**：在 ALFWorld 未見過的評估集上達到可報告的成功率，並畫出相對 LLM 基準的成本–成功率 Pareto 前緣。

## Phase 2：雙人對抗 / 領域應用　⬜（8–12 週）

- **CybORG / CyberBattleSim 紅藍隊自我對弈**：狀態文字化、league training、Elo 曲線（搜尋核心已經支援雙人零和）
- 或 **多跳檢索問答**（HotpotQA / MuSiQue），並發展中文版本（`laya-multilingual`）

**出口條件**：真正的 AlphaZero 式對抗學習曲線（Elo 隨迭代上升），或領域任務上相對基準的顯著提升。

## Phase 3（長期）：LLM 提案 + Laya 價值　⬜

- Lean 4 定理證明（LeanDojo / miniF2F）：tactic 由 LLM 提出，Laya 負責重排序與價值估計
- 一般化的「System 2 提案 / System 1 評估」框架（新的 `Evaluator` 加上提案器）
