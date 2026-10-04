# TODO：目前目標的進度

> 大目標見 [`ROADMAP.md`](ROADMAP.md)。本檔只追蹤**目前的 Phase** 與緊接著的工作。
> 圖例：`[x]` 完成　`[ ]` 未完成　`[!]` 受阻（附原因）

## Phase 0（管線驗證）：✅ 完成，見 [`docs/phase0_report.md`](docs/phase0_report.md)

### 0.1 環境建立
- [x] Python 3.11 venv（uv）+ `laya==0.3.24`、torch 2.14、transformers 5.18、SymPy、pytest（`scripts/setup_env.sh`）
- [x] `mcts-laya download`：已下載 `multilingual`（644 MB）與 `english`（804 MB）到 `models/laya/`
- [x] `mcts-laya make-tiny`：離線用的微型 Laya 相容 checkpoint
- [!] 推送到 GitHub：`kthfan/mcts_laya` 尚未授權給 Claude（403），所有 commit 目前只存在本地分支

### 0.2 模組化程式碼
- [x] `envs`：函數式 `Environment` 介面 + registry；Countdown、一元一次方程（SymPy 驗證）
- [x] `teachers`：Countdown 窮舉、方程 BFS
- [x] `LayaEvaluator`：每個葉節點候選不同的批次推論、依長度排序、LRU cache、選項順序隨機化
- [x] `search`：PUCT（virtual loss 批次）、Gumbel AlphaZero、greedy；雙人零和 backup
- [x] `selfplay`：outcome / root / mix 價值目標、policy 目標銳化、失敗局的 policy 權重
- [x] `LayaTrainer`：soft CE + RLCD、選項順序擴增、溫度校準、凍結 embedding / encoder、gradient checkpointing、輸出 Laya checkpoint
- [x] `AlphaZeroLoop` + YAML 設定 + CLI（`run` / `download` / `make-tiny` / `bench`）
- [x] pytest 31 項

### 0.3 Phase 0 實驗
- [x] 微型模型：代數（弱 warm start、從零開始）三項里程碑全部 PASS
- [x] 微型模型：Countdown（2/3，微型模型學算術的能力不足；已確認是模型限制，不是管線問題）
- [x] 真正的 Laya（CPU 縮小版）：Countdown 三項 PASS（0.93 vs 均勻 0.30）
- [x] 真正的 Laya（CPU 縮小版）：代數 warm start 後達最優解（天花板）
- [x] 真正的 Laya：代數從零開始，4 輪內 greedy 從 0.02 升到 0.71（尚未超過均勻先驗的搜尋）

### 0.4 Phase 0 留下的事項（不阻擋 Phase 1）
- [ ] 在 GPU 上跑完整規模設定（`configs/phase0/{algebra,countdown}.yaml`，評估 100 題）
- [ ] 真權重的代數從零開始版跑到 8 輪以上，確認能超過均勻先驗的搜尋
- [ ] 多 seed 或成對統計檢定（目前的 30 題評估，差距 < 0.05 算雜訊）
- [ ] 比較 `english` 與 `multilingual` checkpoint 的學習效率

## 目前目標：Phase 1a（TextWorld）

### 1a.1 介面與 adapter
- [ ] `Environment.sample_problem(rng, split)`、`sample_problems(rng, n, split)`；pipeline 的評估改用 `split="eval"`
- [ ] `TextWorldEnv`：狀態 = (遊戲, 動作歷史) + 觀測快照；`GameRunner` 用 reset + 前綴重播；遊戲開啟數的 LRU 上限
- [ ] 指令過濾（預設去掉 look / inventory / examine，過濾後為空時退回完整清單）
- [ ] 狀態文字：目標（去掉 TextWorld 開場白）、最近 k 個動作、物品欄、位置描述、上一步結果、剩餘步數
- [ ] reward：成功 = 保底 + γ^步數；失敗 / 逾時 = 0（可選部分得分）

### 1a.2 遊戲池與 teacher
- [ ] 關卡定義 L1 / L2 / L2-goal / L3-goal；`mcts-laya tw-games` 平行產生，manifest 記錄 train / eval
- [ ] `TextWorldTeacher`：`policy_commands` 的第一步為最佳動作，剩餘長度換算價值
- [ ] 微型模型的 tokenizer 語料支援帶參數的環境

### 1a.3 測試
- [ ] 重播的確定性、狀態不可變、過濾、reward、teacher 解題、端到端迴圈

### 1a.4 實驗（真 Laya，CPU 縮小版）
- [ ] L1（完整指示）
- [ ] L2（完整指示）
- [ ] L2-goal（只給目標）
- [ ] L3-goal（只給目標）
- [ ] 報告 `docs/phase1a_report.md`

### 1a.5 之後
- [ ] cooking 關卡（橋接）：需要保留 `examine cookbook`，動作數較多
- [ ] 動作數 > 20：shortlist / 階層式 choice（ALFWorld 也需要）
- [ ] 部分可觀測的記憶：已探索房間與看過的物件摘要

## 視覺化介面（並行工作）
評估：Countdown、代數沒有現成引擎，用 HTML 前端；TextWorld 以引擎的 `textworld.render`（世界狀態 → 地圖 JSON）為基礎；ALFWorld 文字版沿用 TextWorld 做法，3D（AI2-THOR）需要 GPU，列為選配。
- [ ] 搜尋紀錄匯出：每步候選（先驗 / 拜訪 / Q / 策略）、價值、有上限的搜尋樹；環境專屬畫面資料 `render_data`
- [ ] teacher 也當成一種「搜尋」，供同題比較
- [ ] 前端（單一 HTML，純 JS + SVG，深淺色）：對局回放、搜尋樹瀏覽、同題比較、學習曲線
- [ ] 靜態報告：`mcts-laya viz <run_dir>`
- [ ] 即時模式：`mcts-laya serve`（出題、逐步搜尋、自己選步）
- [ ] TextWorld 地圖面板（`textworld.render.load_state`）
- [ ] ALFWorld：文字版沿用；3D 畫面待 GPU

## Phase 1b（ALFWorld）預備
- [ ] 多 episode 並行、跨 episode 合併葉節點批次（GPU 吞吐量）
- [ ] 搜尋樹重用；MuZero Reanalyse
- [ ] ALFWorld 安裝與資料下載、PDDL expert teacher

## 已知觀察
- 預訓練過的 Laya 學得遠比隨機初始化的模型快（代數 150 題 vs 1000 題；Countdown 0.83 vs 0.37）。
- 價值網路弱時，Gumbel（每個候選 1–2 次模擬）不如 PUCT；oracle 價值能讓同樣的搜尋從 0.29 升到 0.93。
- 獎勵設計要讓「晚一點成功」明顯好於失敗，否則樂觀的價值網路會讓搜尋一直拖延。
- 真權重（multilingual）在 4 核 CPU 上：代數 11.7 rows/s、Countdown 4.1 rows/s。完整規模的實驗需要 GPU。
