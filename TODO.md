# TODO：目前目標的進度

> 大目標見 [`ROADMAP.md`](ROADMAP.md)。本檔只追蹤**目前的 Phase** 與緊接著的工作。
> 圖例：`[x]` 完成　`[ ]` 未完成　`[!]` 受阻（附原因）

## 目前目標：Phase 0（管線驗證）

### 0.1 環境建立
- [x] Python 3.11 venv（uv）+ `laya==0.3.24`、torch 2.14、transformers 5.18、SymPy、pytest（`scripts/setup_env.sh`）
- [x] `mcts-laya download`：從 HF 下載 `english` / `multilingual` / `typed-decisions` checkpoint
- [x] `mcts-laya make-tiny`：離線用的微型 Laya 相容 checkpoint
- [!] 在本容器下載真正的 Laya 權重：容器的網路政策擋住 `huggingface.co`（使用者已表示會開放，但截至目前仍是 403）
- [!] 推送到 GitHub：`kthfan/mcts_laya` 尚未授權給 Claude（403），commit 目前只存在本地分支

### 0.2 模組化程式碼
- [x] `envs`：函數式 `Environment` 介面 + registry
- [x] Countdown 環境、窮舉 teacher
- [x] 一元一次方程環境（5 種題型模板、Fraction 精確運算、SymPy 驗證每一步）、BFS teacher
- [x] `LayaEvaluator`：每個葉節點候選不同的批次推論、LRU cache、選項順序隨機化
- [x] PUCT（virtual loss 批次、Dirichlet 噪音、FPU、平手處理）、Gumbel AlphaZero、greedy 基準
- [x] 雙人零和 backup（以 Nim 測試）
- [x] self-play：outcome / root / mix 價值目標、policy 目標銳化、失敗 episode 的 policy 權重
- [x] `LayaTrainer`：soft CE + 選用 RLCD、選項順序擴增、每型別溫度校準、存成 Laya checkpoint
- [x] `AlphaZeroLoop`：warm start → self-play → 訓練 → 校準 → 評估 → (gate)，輸出 `metrics.jsonl` / `summary.md`
- [x] YAML 設定 + `--set` 覆寫 + 未知 key 檢查；CLI：`run` / `download` / `make-tiny` / `bench`
- [x] pytest 30 項（環境正確性、搜尋、Laya 整合、訓練 round-trip、端到端迴圈）

### 0.3 Phase 0 實驗
- [x] 診斷：搜尋本身有效（oracle 價值：Countdown 0.29 → 0.80–0.93）
- [x] 修正：代數 reward 加保底（避免搜尋拖延到步數用完）、self-play 改用 PUCT、policy 目標銳化
- [x] 修正：微型 tokenizer 的數字邊界、Countdown 動作文字加上 `(gap N)`
- [x] 代數 / 微型模型：三個里程碑全部 PASS（見下方結果）
- [ ] Countdown / 微型模型（執行中）
- [ ] 用新 tokenizer 重跑代數 / 微型模型，確保兩個任務使用同一版模型
- [!] 代數與 Countdown / 真正的 Laya（`configs/phase0/{algebra,countdown}.yaml`）：等 HF 開放

### 0.4 Phase 0 收尾
- [ ] 真正的 Laya 權重下跑 `mcts-laya bench`，記錄 rows/s（CPU / GPU）
- [ ] 真正的 Laya 權重下比較 `english` 與 `multilingual` checkpoint
- [ ] 每個評估點用多個 seed 或成對統計檢定（目前差距在 100 題下仍有雜訊）

## 下一步（Phase 1 前置）
- [ ] 多 episode 並行、跨 episode 合併葉節點批次（GPU 吞吐量）
- [ ] 搜尋樹重用（下一步沿用子樹）
- [ ] MuZero Reanalyse（用新網路重算舊軌跡的目標）
- [ ] 凍結 encoder / LoRA 的訓練模式，降低每輪迭代成本
- [ ] 有狀態模擬器 adapter（save/restore 或動作重播），以及 ALFWorld / WebShop 環境選型
- [ ] 選項數 > 20 的處理：shortlist / 階層式 choice / progressive widening

## 已知觀察
- 微型模型（隨機初始化）在價值頭上學得遠不如 policy 頭；價值不準時 Gumbel（每個候選只有 1–2 次模擬）不如 PUCT。
- 16 次模擬的拜訪分佈很平坦，直接當 policy 目標會稀釋 teacher 教出的先驗，所以使用 τ = 0.5 銳化。
- 從零訓練的模型讀不了沒有邊界的逐位數字；真正的 Laya 用 BPE tokenizer，應該不受影響，但仍需實測（README 也提到 Laya 的數字推理偏弱）。
