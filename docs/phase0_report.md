# Phase 0 報告：管線驗證

> 日期：2026-10-03　｜　硬體：雲端容器，4 核 CPU、15 GB RAM、無 GPU　｜　Laya `0.3.24`
> 原始數據：[`docs/results/phase0/`](results/phase0/)（每組實驗有 `*.md` 摘要與 `*.metrics.jsonl`）

## 結論

**Phase 0 的目標已經達成。** 整個 AlphaZero / Expert Iteration 迴圈都能端到端運作，而且在兩個任務上都證明會讓模型進步：

1. 用**真正的 Laya**（`laya-multilingual`）時，兩個任務都在少量 teacher 資料下學會，搜尋也明顯勝過不用 Laya 的基準：
   - Countdown：Laya + PUCT16 成功率 **0.93**，均勻先驗的 PUCT16 只有 0.30
   - 代數：只用 150 題 teacher 資料就達到最優解（reward 0.776，理論上限約 0.777）
2. **純 self-play、完全不用 teacher** 也會學起來。在代數任務上，不搜尋的網路本身：
   - 微型模型：0.00 → 0.71
   - 真正的 Laya：0.02 → 0.71（只跑了 4 輪）
3. 過程中找到並修正了 5 個會讓迴圈失效的問題（見「過程中修正的問題」），它們對 Phase 1 有直接參考價值。

保留事項：
- 真權重的實驗受限於 CPU，只跑了縮小版：評估 30 題、self-play 每輪 16 局。差距小於約 0.05 時屬於雜訊範圍。
- 完整規模的設定（`configs/phase0/{algebra,countdown}.yaml`）留給 GPU 執行。

## 里程碑定義

每個任務都要通過：
1. Laya+MCTS 的評估 reward 隨 self-play 迭代上升
2. Laya+MCTS 勝過 Laya greedy（不搜尋）
3. Laya+MCTS 勝過同預算、均勻先驗的 MCTS

評估指標：
- `reward`：Countdown 是成功率；代數是 `0.5 + 0.5·0.9^步數`，未解出為 0
- 評估時固定用 PUCT 16 次模擬（`puct16`），也會記錄 Gumbel16

## 結果總表

| 實驗 | 模型 | 起點 | 結果（PUCT16 reward） | greedy | 均勻 PUCT16 | 里程碑 1 / 2 / 3 |
|---|---|---|---|---|---|---|
| Countdown | **Laya** | teacher 150 題 + 1 輪 | 0.17 → 0.83 → **0.93** | 0.70 | 0.30 | ✅ ✅ ✅ |
| 代數 | **Laya** | teacher 150 題 + 4 輪 | 0.35 → 0.776 → 0.777 | 0.776 | 0.727 | ✅ ✅* ✅ |
| 代數 | **Laya** | **從零開始**，4 輪 | 0.35 → 0.68 → 0.71 | 0.02 → 0.71 | 0.727 | ✅ ❌ ❌ |
| 代數 | 微型 | 弱 teacher（150 題）+ 6 輪 | 0.73 → 0.75 | 0.60 → 0.72 | 0.730 | ✅ ✅ ✅ |
| 代數 | 微型 | **從零開始**，8 輪 | 0.07 → 0.54 → 0.74 | 0.00 → 0.61 | 0.730 | ✅ ✅ ✅ |
| 代數 | 微型 | 強 teacher（1000 題）+ 6 輪 | 0.769 → 0.772 | 0.776 | 0.730 | ✅ ❌* ✅ |
| Countdown | 微型 | teacher 1000 題 + 6 輪 | 0.22 → 0.37 → 0.35 | 0.15 | 0.30 | ❌ ✅ ✅ |
| Countdown | 微型 | 同上，self-play 用 64 次模擬 | 0.19 → 0.14 | 0.05 | 0.30 | ❌ ✅ ❌ |

\* 天花板效應：warm start 之後已經是最優解（代數最優約 0.777、5.6 步），搜尋和 greedy 都沒有空間再提高，所以 ✅/❌ 只反映 0.001 等級的雜訊。

### 觀察
- **預訓練非常重要。** 同樣的 Countdown 設定下，真正的 Laya 用 150 題就到 0.83，微型模型用 1000 題也只到 0.37。代數也一樣：Laya 用 150 題的 held-out policy top-1 是 0.97，微型模型用 1000 題是 0.81。
- **從零開始是真正的 AlphaZero 證據。** 網路一開始幾乎不會解題（greedy ≈ 0），self-play 第 1 輪只有 25% 的局成功，第 3 輪就 100%。用真權重從零開始時，4 輪內還沒追上均勻先驗的搜尋，這是迭代次數太少的問題（微型模型版到第 8 輪才超過）。
- **搜尋的效益取決於價值網路的品質。** 換成 oracle 價值時，同樣的搜尋成功率從 0.29 升到 0.80–0.93；價值網路弱時，PUCT 比 Gumbel 穩定。
- **吞吐量**（`laya-multilingual`，CPU 4 核，依長度排序批次）：代數 11.7 rows/s、Countdown 4.1 rows/s；`laya`（英文 421M）約慢一半。

## 過程中修正的問題

| 問題 | 症狀 | 修正 |
|---|---|---|
| 代數 reward `0.9^steps` 讓晚解出的價值接近 0 | 「均勻先驗 + Laya 價值」幾乎全失敗：價值網路的樂觀估計高過終局，搜尋一直拖延到步數用完 | `success_floor`：reward = 0.5 + 0.5·γ^steps，解出一定明顯好於失敗 |
| 16 次模擬的拜訪分佈很平坦 | self-play 把 teacher 教出的尖銳先驗稀釋掉 | `policy_target_temperature: 0.5`；失敗局只訓練 value（`failed_policy_weight: 0`） |
| PUCT 在模擬次數 = 批次大小時拜訪數平手 | 選到第一個動作（測試抓到） | 平手時改比 Q 值，再比 prior |
| Gumbel 根節點的平均值被差的動作拉低 | `root_value` 偏低 | 回報改進後 policy 加權的 completed Q |
| 微型 tokenizer 丟掉數字邊界 | `4, 50, 100` 變成 `4 5 0 1 0 0`，Countdown 完全學不到（CE 等於均勻） | Metaspace 詞首標記；Countdown 動作加上 `(gap N)` |
| 評估器把短的 value row 補齊到長的 policy row | Countdown 只有 1.6 rows/s | 依長度排序分批 → 4.1 rows/s（代數 7.4 → 11.7） |
| 全量微調 mmBERT 在長序列上記憶體不足 | Countdown 訓練被 OOM kill | `freeze_embeddings` + `gradient_checkpointing`，batch 4 |
| 沒有設 torch 亂數種子 | 同一設定 warm start 結果 0.19 vs 0.37 | 迴圈開始時 `torch.manual_seed(seed)` |

## 重現

```bash
scripts/setup_env.sh --download                       # 環境 + laya-multilingual
.venv/bin/python -m pytest                            # 31 項測試
.venv/bin/mcts-laya run configs/phase0/countdown_cpu_smoke.yaml   # 真權重，CPU 約 1.5 小時
.venv/bin/mcts-laya run configs/phase0/algebra_cpu_smoke.yaml     # 真權重，CPU 約 1.5 小時
.venv/bin/mcts-laya run configs/phase0/algebra_cpu_smoke.yaml --set teacher.enabled=false output_dir=runs/phase0/algebra-laya-cpu-scratch
.venv/bin/mcts-laya run configs/phase0/algebra_tiny.yaml          # 微型模型，離線
.venv/bin/mcts-laya run configs/phase0/algebra.yaml               # 完整規模（建議 GPU）
```
