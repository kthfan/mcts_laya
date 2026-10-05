# mcts_laya

用 [Laya](https://github.com/NandhaKishorM/laya)（Convai Innovations 的非自迴歸決策模型）當 policy / value 網路，搭配蒙地卡羅樹搜尋，做 AlphaZero / Expert Iteration 式的自我改進。

- 研究與候選問題評估：[`candidate_problems.md`](candidate_problems.md)
- 大目標：[`ROADMAP.md`](ROADMAP.md)
- 目前進度：[`TODO.md`](TODO.md)
- Phase 0 結果：[`docs/phase0_report.md`](docs/phase0_report.md)

## 環境建立

```bash
scripts/setup_env.sh               # 建 .venv、安裝本套件（含 laya==0.3.24、torch、sympy、pytest）
scripts/setup_env.sh --download    # 再下載 laya-multilingual 權重到 models/laya/multilingual
# 或分開執行：
.venv/bin/mcts-laya download --checkpoint multilingual english --out models/laya
```

權重從 `huggingface.co/convaiinnovations/laya` 下載，需要能連上 Hugging Face。連不上時，可以用微型的隨機初始化模型（磁碟格式與 Laya 完全相同）在 CPU 上開發與測試：

```bash
.venv/bin/mcts-laya make-tiny --out models/tiny
```

## 執行

```bash
.venv/bin/python -m pytest                                        # 單元與整合測試（用微型模型，約 20 秒）
.venv/bin/mcts-laya run configs/phase0/countdown_cpu_smoke.yaml  # Phase 0：真權重縮小版（CPU 約 1.5 小時）
.venv/bin/mcts-laya run configs/phase0/countdown_tiny.yaml        # Phase 0：Countdown，微型模型，離線
.venv/bin/mcts-laya run configs/phase0/algebra_tiny.yaml          # Phase 0：解一元一次方程，微型模型
.venv/bin/mcts-laya run configs/phase0/countdown.yaml             # 同上，用真正的 Laya 權重
.venv/bin/mcts-laya run configs/phase0/algebra.yaml --set iterations=3 search.params.num_simulations=32
.venv/bin/mcts-laya bench --checkpoint models/laya/multilingual --env countdown   # 評估器吞吐量
```

每個實驗會在 `output_dir` 寫出：`config.yaml`（完整設定）、`metrics.jsonl`（每個階段一筆紀錄）、`summary.md`（學習曲線、基準比較、Phase 0 里程碑檢查）、`teacher_samples.jsonl`，以及 `checkpoints/`（Laya 相容格式，可以直接 `laya.load`）。

## GPU 實驗

TextWorld self-play 消融（A：只學有效率的成功局、B：只訓練 value、C：更深的搜尋、D：DAgger 上限參考）的完整步驟見 [`docs/gpu_experiments.md`](docs/gpu_experiments.md)：

```bash
.venv/bin/python scripts/run_ablation.py configs/ablation/selfplay_textworld.yaml --only L2-goal --seeds 0
.venv/bin/python scripts/summarize_ablation.py runs/ablation/selfplay_textworld
```

## 視覺化

```bash
.venv/bin/mcts-laya viz --run runs/phase0/countdown-laya-cpu-smoke   # 靜態 HTML 報告（對局回放、搜尋樹、同題比較、學習曲線）
.venv/bin/mcts-laya serve --run runs/phase1a/l1-laya-cpu             # 即時模式：http://127.0.0.1:8765
```

說明與各環境的評估見 [`docs/visualization.md`](docs/visualization.md)，已產生的報告在 [`docs/viz/`](docs/viz/)。

## 架構

```
src/mcts_laya/
  envs/          Environment 介面（函數式：step 回傳新狀態，搜尋不需要存檔/還原）
                 countdown.py   Countdown 數字遊戲
                 algebra.py     一元一次方程逐步改寫（Fraction 精確運算，SymPy 驗證）
  evaluators/    Evaluator 介面：(states, actions) -> (priors, value)
                 laya_evaluator.py  Laya 批次評估器（每個葉節點候選不同，自行組 row 共用 forward）
                 base.py            uniform / rollout 基準
  search/        tree.py（節點、展開、backup，支援雙人零和）
                 puct.py（AlphaZero PUCT + virtual loss 批次）
                 gumbel.py（Gumbel AlphaZero，Sequential Halving；greedy 無搜尋基準）
  selfplay/      play_episode / episode_to_samples（outcome / root / mix 價值目標）
  teachers/      冷啟動用的精確解題器（Countdown 窮舉、方程 BFS）
  training/      Sample（純文字 + 軟目標，JSONL）、ReplayBuffer、LayaTrainer
                 （soft CE + 選用 RLCD、選項順序擴增、溫度校準、存成 Laya checkpoint）
  models/        download.py（HF 下載）、tiny.py（微型相容模型）
  viz/           搜尋紀錄匯出、靜態報告（viz）、即時伺服器（serve）、單檔前端（static/）
  laya_io.py     局面 <-> Laya 問題（policy = choice、value = noul）的唯一轉換點
  pipeline.py    AlphaZeroLoop：warm start -> [self-play -> 訓練 -> 校準 -> 評估 -> gate]*
  config.py      YAML -> dataclass，支援 --set a.b=c 覆寫，未知的 key 直接報錯
  registry.py    ENVIRONMENTS / EVALUATORS / SEARCHERS / TEACHERS 註冊表
```

### 一個局面怎麼變成 Laya 的輸入

| | Laya 問題型別 | 內容 |
|---|---|---|
| policy | `choice` | 選項 = 候選動作文字（標籤 A, B, C…），輸出 P(s,·) |
| value | `noul` | 「這個狀態會成功嗎？」，V(s) = 2·P(true) − 1 |

訓練目標：policy row 用 MCTS 的拜訪 / 改進策略分佈，value row 用 [1−p, p]（p = (z+1)/2），兩者都用 soft cross-entropy，可選加上 Laya 的 RLCD policy-gradient 項。

## 擴充

**新環境**：繼承 `Environment`（或 `SingleAgentEnvironment`），實作 `sample_problem / legal_actions / step / is_terminal / terminal_value(或 terminal_reward) / state_text / action_text`，並覆寫三段 Laya 問句（`policy_instruction`、`value_instruction`、`value_criteria`），最後用 `@ENVIRONMENTS.register("name")` 註冊，YAML 的 `env.name` 就能使用。有狀態的模擬器（例如 Jericho、瀏覽器）在 `step` 內用模擬器自己的 save/restore。雙人遊戲覆寫 `current_player`，backup 會自動換邊（見 `tests/test_search.py` 的 Nim）。

**新搜尋法**：繼承 `Searcher`，實作 `search(state, rng, add_noise) -> SearchResult`，用 `@SEARCHERS.register` 註冊。

**新評估器**（例如 LLM 提案 + Laya 評分、ONNX 推論）：繼承 `Evaluator`，用 `@EVALUATORS.register` 註冊。

**冷啟動 teacher**：繼承 `OracleTeacher`，實作 `analyse(state, actions) -> (value, best_indices)`，用 `@TEACHERS.register("<env name>")` 註冊。
