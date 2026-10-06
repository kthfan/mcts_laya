# GPU 實驗步驟：TextWorld self-play 消融（A / B / C / D）

## 要回答的問題

在 CPU 縮小版實驗裡，warm start（模仿 teacher）之後的 self-play 沒有讓 Laya 進步。即使加了 gate，L2-goal 的三輪 self-play 也全部被判定退步而還原（PUCT16 reward 0.754 → 0.611 / 0.534 / 0.694）。這組實驗要找出**哪一種 self-play 目標能讓 agent 超越它的 warm start**。

| 變體 | 設定 | 檢驗的假設 |
|---|---|---|
| `control` | 成功局的搜尋結果當 policy 目標（目前的做法），self-play 16 次模擬 | 對照組 |
| `A-efficient` | 只學「相對於自己紀錄有效率」的成功局：步數 ≤ 1.3 × 自己在該題最少的步數，且 reward 在本輪成功局的前 50% | 退步來自模仿繞遠路的成功局 |
| `B-value-only` | self-play 只訓練 value，policy 保留 warm start 學到的 | 退步來自 policy 被模糊目標稀釋；更準的 value 能讓搜尋變好 |
| `C-sims48` | self-play 用 48 次模擬（評估仍是 16 次） | 搜尋太淺，目標沒有比網路強 |
| `AC-efficient-sims48`、`BC-value-only-sims48` | A / B 再加上 48 次模擬 | 兩種修正可以疊加 |
| `D-dagger` | teacher 為 self-play 走過的每個狀態標註最佳動作與價值 | **不是**自我改進，是模仿學習；當作「如果有完美標註能到多好」的上限參考 |

所有變體共用的設定都已經寫在各等級的設定檔裡：gate（每輪 PUCT16 reward 下降超過 0.02 就還原）、policy 目標溫度 0.25、保留全部 teacher 資料，以及每個 seed 相同的評估遊戲（成對比較）。

程式碼位置：`src/mcts_laya/selfplay/targets.py`（A、B、D），`configs/ablation/selfplay_textworld.yaml`（變體定義）。

## 0. 硬體與時間

- **GPU**：16 GB 以上（T4、L4、A10、A100 皆可）。T4 / V100 會用 fp16，訓練器已經加上 `GradScaler`；Ampere 以上會用 bf16。
- **CPU**：4 核以上。TextWorld 引擎的重播在 CPU 上跑，也會占用時間。
- **磁碟**：約 10 GB（venv 約 6 GB，multilingual 權重 0.65 GB，每個 run 的 checkpoint 0.65 GB）。
- **時間（估計，請以第 3 步的試跑為準）**：L2-goal 一個 run（8 輪）在 T4 上約 2–3 小時，在 A100 上約 1 小時；`*-sims48` 變體的 self-play 時間約 3 倍。L2-goal 的 7 個變體 × 3 個 seed 共 21 個 run。建議依第 4 步分階段進行。

## 1. 取得程式碼與建立環境

```bash
git clone https://github.com/kthfan/mcts_laya.git && cd mcts_laya
git checkout claude/laya-mcts-project-planning-oj7zb4

# 先確認 PyTorch 的 CUDA 版本和驅動相容。預設會從 PyPI 安裝 torch（目前是 CUDA 13 版本）；
# 驅動較舊的話，先在 venv 裡裝對應版本，例如：
#   python -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu124
scripts/setup_env.sh --download            # 建 .venv、安裝本套件、下載 laya-multilingual
.venv/bin/pip install -e ".[dev,textworld]"  # TextWorld
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`cuda.is_available()` 一定要是 `True`。

## 2. 確認環境與產生遊戲池

```bash
.venv/bin/python -m pytest                   # 約 1–2 分鐘，應該全部通過
.venv/bin/mcts-laya bench --checkpoint models/laya/multilingual --env algebra --device cuda --batch 64 --positions 256
# 記下 rows_per_second（CPU 上是 11.7；GPU 應該高出一到兩個數量級）

.venv/bin/mcts-laya tw-games --level L2-goal L3-goal C1 --train 150 --eval 40 --out data/textworld --workers 8
```

遊戲池由固定的 seed 產生，和 CPU 實驗用的是同一批遊戲（train seed 0–149、eval seed 1000000–1000039）。

## 3. 試跑（約 15–30 分鐘）

先用 1 輪確認整條管線在 GPU 上正常，並量出每輪需要的時間：

```bash
.venv/bin/python scripts/run_ablation.py configs/ablation/selfplay_textworld.yaml \
    --only L2-goal --variants control D-dagger --seeds 0 --extra iterations=1 --output-root runs/ablation/smoke
cat runs/ablation/smoke/L2-goal/control-s0/summary.md
```

檢查以下幾點：
- 兩個 run 都 `exit 0`。
- `summary.md` 中 warm start 的 PUCT16 成功率約 0.9 以上（CPU 版是 0.95）。
- `Self-play` 表格的 `rows_per_second`，以及 `run.log` 裡每個階段的時間。用這些數字推算正式實驗要多久。

**如果記憶體不足**：在 `configs/ablation/selfplay_textworld.yaml` 的 `gpu_overrides` 把 `train.config.batch_size` 改成 8。**24 GB 以上的卡**可以加上 `train.config.gradient_checkpointing=false`，加快訓練。

## 4. 正式實驗（分階段）

```bash
# 第一階段：每個變體先跑 seed 0（L2-goal，7 個 run）
.venv/bin/python scripts/run_ablation.py configs/ablation/selfplay_textworld.yaml --only L2-goal --seeds 0

# 第二階段：補 seed 1、2（可以只挑第一階段有希望的變體，加上 control 和 D）
.venv/bin/python scripts/run_ablation.py configs/ablation/selfplay_textworld.yaml --only L2-goal --seeds 1 2
#   例如：--variants control A-efficient C-sims48 D-dagger --seeds 1 2

# 第三階段：把最好的變體（和 control）延伸到 L3-goal、C1
.venv/bin/python scripts/run_ablation.py configs/ablation/selfplay_textworld.yaml --only L3-goal C1 \
    --variants control <最好的變體> --seeds 0 1 2
```

- **平行對局（自動開啟）**：模型在 GPU 上時，每個實驗會自動啟動「CPU 核心數 − 1」個（最多 32 個）actor 程序，同時進行多局 self-play 與評估。每個 actor 自己處理 TextWorld 引擎、搜尋樹和 tokenization，主程序只把所有 actor 的葉節點請求合併成一批送進 GPU（AlphaZero 的做法）。`summary.md` 的 Self-play 表格會多出三欄：`workers`；`rows_per_batch`（每次 GPU 呼叫的列數）；`model_busy`（GPU 呼叫占總時間的比例，接近 1 代表再加 actor 也不會更快）。可以用 `--extra parallel.workers=N` 指定 actor 數量，`parallel.workers=0` 改回單局循序執行。每個 actor 約占 0.5–0.8 GB 記憶體（RAM）。
- **同時跑多個實驗**：`--jobs N` 同時跑 N 個實驗、共用同一張 GPU，每個實驗分到各自的一組 CPU 核心（每個實驗的 actor 數量也跟著變成「分到的核心數 − 1」）。有了平行對局後，單一實驗已經能用滿所有核心，`--jobs` 主要用來讓訓練階段（只用 GPU）和其他實驗的對局重疊。建議先用 `--jobs 2`。每個實驗約占 6–10 GB 顯存：H100 80 GB 最多可以跑約 6 個，用 `nvidia-smi` 確認顯存和使用率後再調整（上限大約是 CPU 核心數，以及顯存 ÷ 10 GB）。例如：
  ```bash
  .venv/bin/python scripts/run_ablation.py configs/ablation/selfplay_textworld.yaml --only L2-goal --seeds 0 --jobs 2
  ```
- 腳本可以續跑：已完成的 run（目錄裡有 `done.json`）會跳過，中斷後重下同一行指令即可。中斷中的 run 會從頭開始。
- 長時間執行建議包在 `tmux` 或 `nohup ... &` 裡。
- CPU：預設使用所有核心。需要保留核心給其他工作時，可以用 `--cpus N` 限制（例如 `run_ablation.py ... --cpus 6`；TextWorld 重播、tokenizer、torch 執行緒都包含在內）。
- 進度：`tail -f runs/ablation/selfplay_textworld/L2-goal/<變體>-s<seed>/run.log`。每 30 秒會有一行目前階段的進度，例如 `[it 3/8 self-play] 44/64 (69%) episodes, 2.08 episodes/s, elapsed 0:21, eta 0:09 | success=0.791, moves=7.74`（間隔可用 `MCTS_LAYA_PROGRESS_INTERVAL=10` 調整）。同目錄的 `summary.md` 每輪更新。

## 5. 彙整與判讀

```bash
.venv/bin/python scripts/summarize_ablation.py runs/ablation/selfplay_textworld
#   -> runs/ablation/selfplay_textworld/summary.md（每個變體跨 seed 的平均 ± 標準差）與 runs.csv（每個 run）
```

表格中的欄位：
- `final reward`、`best reward`：最後一輪與最佳一輪的 PUCT16 評估 reward。有 gate，所以最終權重不會比最佳權重差太多。
- `gain vs warm start`：最後一輪減掉 warm start 的 reward。**這是主要指標**。
- `greedy final`：不搜尋時網路本身的表現，用來看網路有沒有真的變強。
- `final − control (paired)`：每個 seed 的 final reward 減掉 control 同一個 seed 的 final reward，再取平均。warm start 依 seed 而異，成對比較可以抵消這部分差異，3 個 seed 時比單純比平均可靠。
- 只計入有 `done.json` 的 run；沒跑完的會另外列出（重下同一行 `run_ablation.py` 指令會補跑）。重啟過的 run 只採用最後一次的紀錄。
- `gate accepted`：每個 seed 有幾輪 self-play 被 gate 接受（例如 `3/8`）。

判讀方式（建議）：
1. **有改善**：某個變體 3 個 seed 的平均 `gain vs warm start` ≥ +0.03，且大多數 seed 至少有 2 輪被 gate 接受。
2. 拿各變體和 `D-dagger` 比較：D 的提升幅度代表「目標完美時」的上限，A / B / C 越接近 D 越好。
3. 如果 A、B、C 都沒有改善，而 D 有，代表目前的 self-play 目標品質不足，下一步考慮更強的搜尋（例如 128 次模擬，或 Gumbel 搭配更好的 value）。如果連 D 都沒有改善，代表 warm start 已經接近這個等級的上限，應該改用更難的等級（L3-goal、C1）來比較。

## 6. 要傳回的結果

```bash
tar czf ablation_results.tgz runs/ablation/selfplay_textworld/summary.md runs/ablation/selfplay_textworld/runs.csv \
    runs/ablation/selfplay_textworld/*/*/{summary.md,metrics.jsonl,config.yaml}
```

不需要傳 checkpoint。如果想看某個 run 的對局細節，可以產生視覺化報告（約幾分鐘）：

```bash
.venv/bin/mcts-laya viz --run runs/ablation/selfplay_textworld/L2-goal/A-efficient-s0 --problems 4 --device cuda
```

把 `ablation_results.tgz` 放進 repo（例如 `docs/results/ablation/`）或貼回對話，我就能接著分析並決定下一步。
