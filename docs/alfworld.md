# Phase 1b：ALFWorld——評估、計畫與實作

> 2026-10-08。程式碼：`src/mcts_laya/envs/alfworld_env.py`、`src/mcts_laya/teachers/alfworld.py`、`configs/alfworld/base.yaml`、`configs/ablation/alfworld.yaml`、`tests/test_alfworld.py`。

## 1. ALFWorld 是什麼

ALFWorld（Shridhar et al., 2021）把 ALFRED 的 3D 家居場景轉成**純文字**的 TextWorld 遊戲。每一局是一個房間（廚房、臥室、客廳、浴室）和一個任務，例如「put a clean ladle in countertop」。玩家只能看到自己去過的地方：

```
You are in the middle of a room. Looking quickly around you, you see a cabinet 18, ..., a fridge 1, a sinkbasin 1, ...
Your task is to: put a clean ladle in countertop.
> go to fridge 1          -> The fridge 1 is closed.
> open fridge 1           -> You open the fridge 1. The fridge 1 is open. In it, you see a apple 2, a cup 2, ...
> take apple 2 from fridge 1
> cool apple 2 with fridge 1
```

六種任務：

| 任務 | 意思 | valid_unseen 局數 |
|---|---|---|
| pick_and_place_simple | 找到物品，放到指定位置 | 24 |
| look_at_obj_in_light | 拿著物品，打開檯燈 | 18 |
| pick_clean_then_place_in_recep | 找到物品，在水槽洗乾淨，再放好 | 31 |
| pick_heat_then_place_in_recep | 找到物品，用微波爐加熱，再放好 | 23 |
| pick_cool_then_place_in_recep | 找到物品，用冰箱冷卻，再放好 | 21 |
| pick_two_obj_and_place | 找到**兩個**同類物品，都放到指定位置 | 17 |

標準評估集：**valid_unseen 134 局**（沒看過的房間，論文通常報這個）與 **valid_seen 140 局**（看過的房間、新的物品擺放）。訓練集 3,553 局。主要指標是**成功率**（50 步內完成）。

和 Phase 1a TextWorld 的差別：
- **部分可觀察，而且要搜尋**：物品藏在抽屜、櫃子、冰箱裡，不打開就看不到。TextWorld 的 L2-goal / L3-goal 也只給最終目標，但房間少、物品少。
- **動作多**：每步平均 23–28 個合法指令，最多 63 個（TextWorld 約 10–22 個）。
- **步數長**：上限 50 步；最佳解 4–10 步，但不知道物品位置時，要多走很多步。

## 2. 評估：可不可行？

在這個容器上實測（4 核 CPU，沒有 GPU）：

| 項目 | 結果 | 對設計的影響 |
|---|---|---|
| 安裝 | `pip install alfworld==0.4.2` 只需要 `textworld[pddl]`（會編譯 fast-downward，約 1 分鐘），不需要 3D 引擎 AI2-THOR | 新增 `.[alfworld]` 安裝選項 |
| 資料 | 只需要 TextWorld-PDDL 版的遊戲檔（36 MB 下載、350 MB 解壓），不需要 `alfworld-download` 的 2.3 GB（含影像偵測模型） | `mcts-laya alfworld-data` 只下載這一份 |
| 每一步的引擎時間 | 原本 52–125 ms（廚房較慢）。**瓶頸是 TextWorld 的文字產生器**：每一步都用 TatSu 重新解析同樣的文法模板，並深拷貝整個上下文 | 快取解析結果、改成淺拷貝（`_speed_up_pddl_textgen`）→ 約 15 ms；兩種版本對同一串指令產生的文字逐字相同（測試比對） |
| 搜尋需要「回到樹上的任一節點」 | TextWorld L 關卡用「reset + 重播歷史」；ALFWorld 的 reset 要 0.35 秒（每次重新翻譯 PDDL），不可行 | **直接還原狀態**：PDDL 狀態 = 每個 SAS 變數的值（0.5–2 KB）。把這些值寫成規劃任務的初始狀態重新載入，約 20 ms；96 + 60 個隨機狀態的還原結果與依序執行完全相同 |
| 最佳規劃器（teacher 用） | 中位數 0.02 秒，但長尾很重：p99 33 秒、最長 **533 秒**，慢的幾乎都在**遊戲一開始**；計畫長度 4–10 步；從還原的狀態規劃，結果和依序執行一致（15/15） | 開局改用遊戲檔內建的 walkthrough（25/25 局合法且能解完）；沿著計畫走時重用剩下的計畫；每次規劃最多 60 秒；teacher 資料用多個程序平行產生。48 局 teacher：規劃 461 次 → 123 次，時間 365 → 45 秒（每程序 12 局），資料完全相同 |
| ALFWorld 內建的手寫 expert | 只靠觀察找物品；valid_seen 140 局成功率 **0.84**，其中 pick_two 只有 **0.21**（常在兩個位置之間來回） | 不適合當主要 teacher（見 §3 的決策） |
| Laya 的輸入長度 | 狀態文字（含記憶）平均 275 tokens、最長 561；所有指令選項平均 225、最長 608 tokens；合計平均約 500、最長約 1170 | `max_len` 1280、`head_max_len` 640（Laya 的 encoder 是 ModernBERT，支援 8192） |
| CPU 上的 Laya | 4 核 CPU 每秒 1.3 列 | **只能在 GPU 上跑**；CPU 只用於測試與除錯 |

**結論：可行。** 主要的技術風險（每步太慢、搜尋無法分支、選項太長）都已解決。剩下的不確定性在學習本身：ALFWorld 需要「有系統地搜尋房間」，這是 TextWorld 關卡沒有測到的能力。

## 3. 決策（2026-10-08 與使用者確認）

| 問題 | 決定 | 理由 |
|---|---|---|
| warm start 的 teacher | **PDDL 最佳規劃器** | 和 TextWorld 的做法相同（在 L2-goal、L3-goal 上，self-play 在此基礎上帶來 +0.12、+0.22）。永遠成功、4–10 步；可以從任何狀態規劃，所以支援探索中的偏離與 DAgger。缺點：它知道藏起來的物品在哪裡，模型只能學到「這類物品通常在哪裡」，「如何搜尋」要靠 self-play |
| gate 與測試集 | gate 用**從訓練集保留的 60 局**；**valid_unseen（134）為主要測試集，valid_seen（140）也報告** | 兩個標準評估集在訓練中完全不碰，結果可以直接和文獻比較 |
| 範圍 | **六種任務全部** | 標準設定，同時報告各任務的成功率 |
| 和 LLM agent 比較（ReAct 等） | **延後** | 先讓 Laya + MCTS 在 ALFWorld 上有結果；之後再比較成本、延遲與成功率 |

## 4. 實作

### 4.1 環境（`envs/alfworld_env.py`）

- **狀態**：`AWState(遊戲, 動作歷史)` + 觀察（回饋文字、合法指令、是否完成、引擎快照）+ 記憶。和 TextWorld 一樣是函數式的：`step` 回傳新狀態，搜尋可以從任何節點分支。初始狀態的觀察延遲到第一次使用時才載入：抽題的主程序不必載入遊戲，實際玩的 actor 才載入。
- **快照**：前 4 個位元組是該遊戲 PDDL 翻譯結果的指紋，其餘是每個變數的值。在主程序建立、在 actor 還原的狀態必須來自相同的翻譯；實測不同 `PYTHONHASHSEED` 的程序翻譯結果相同，指紋是額外的保險（不同就明確報錯）。
- **引擎**：每個遊戲一個 `GameRunner`。下一步的起點如果不是 runner 目前所在的狀態，就從該狀態的快照還原（約 20 ms），不重播。觀察結果有 LRU 快取。
- **runner 重複使用**：TextWorld 每個環境會載入一份 fast-downward 函式庫的副本（32 MB）。卸載會讓下一次載入當機，每個遊戲各留一份又會耗盡 actor 的記憶體上限，所以最多開 `max_open_games`（8）個 runner；需要新遊戲時，把它載入最久沒用的 runner。
- **記憶（狀態文字）**：只用玩家**看到過**的資訊，從每一步的回饋文字解析：

  ```
  Task: put a cool tomato in microwave.
  You are at: drawer 1.
  Holding: nothing.
  Seen: stoveburner 3: kettle 2; diningtable 1: apple 2, bread 1, ..., tomato 2; countertop 1: nothing.
  Closed, not opened yet: fridge 1, drawer 3, cabinet 7.
  Not checked yet: cabinet 1, cabinet 4, ..., microwave 1, sinkbasin 1, toaster 1.
  Recent actions: go to cabinet 15; go to countertop 1; go to cabinet 3; go to drawer 1.
  Last result: You arrive at drawer 1. The drawer 1 is closed.
  Steps left: 35.
  ```

  「Seen」記錄每個去過的容器裡有什麼（拿走、放入時會更新），「Holding」標示物品狀態（hot / cold / clean），「Not checked yet」列出還沒去過的地方，這是搜尋時最需要的資訊。
- **指令過濾**：去掉 look、inventory、examine、help（記憶已經涵蓋這些資訊）。
- **reward**：成功 = 0.5 + 0.5 · 0.95^步數（50 步內的解法仍有差別：10 步 0.80、30 步 0.61），失敗 = 0。主要報告的指標仍是成功率。
- **分割**：`train`（3,493）、`val`（60，gate 用）、`valid_seen`（140）、`valid_unseen`（134）；`eval` = `eval_set`（預設 valid_unseen）。要求的局數不少於整個分割時，就用整個分割（每局一次）。
- **任務類型**：`env.category(state)`；評估紀錄多了 `by_category`（各任務的成功率），`summary.md` 有「Success by category」表。

### 4.2 Teacher（`teachers/alfworld.py`）

從目前狀態取得最佳計畫：第一步是最佳動作，剩下的步數換算價值。和 TextWorld 一樣以 30% 機率隨機偏離，讓價值網路看到較差的狀態。

規劃器偶爾很慢（見 §2），所以：
- 開局用遊戲檔裡的 walkthrough（產生遊戲時由同一個規劃器算出），不重新規劃；
- 照著計畫走時，下一個狀態的計畫就是「剩下的部分」（動作是確定性的，最佳計畫的後半段仍然最佳），只有隨機偏離之後才重新規劃；
- 每次規劃最多 `plan_timeout_s`（60 秒）：規劃在另一個執行緒跑（ctypes 呼叫時會釋放 GIL），逾時就放棄那個引擎、結束這一局（保留已產生的樣本）；
- **`teacher.workers`**（`auto` = 每個空閒核心一個程序，最多 32 個）平行產生，每個程序用自己的 seed 抽遊戲、走一局。同一個 seed 的實驗之間，teacher 資料相同。

### 4.3 Pipeline 的通用擴充

- `eval.extra_splits`：在 warm start 與最終保留的權重上，額外評估其他測試集（ALFWorld：valid_seen）。summary.md 有「Other test splits」表。
- `by_category`：環境有 `category()` 時，評估紀錄附上各類別的成功率。

### 4.4 設定（`configs/alfworld/base.yaml`）

teacher 500 局、warm start 2 個 epoch；8 輪 self-play，每輪 128 局；PUCT 16 次模擬；policy 目標銳化 0.25；gate 容許 0.02；不跑隨機 rollout 基準（每次模擬要玩到 50 步，太貴），保留均勻先驗基準。

### 4.5 測試（`tests/test_alfworld.py`，4 個小遊戲，共 40 KB）

manifest 與分割、記憶解析、快照還原與依序執行一致、runner 重複使用不影響結果、加速前後文字逐字相同、teacher 能解題、完整 pipeline（平行 teacher + 平行 actor + 額外測試集 + 各類別成功率）。

## 5. 如何執行（GPU）

```bash
.venv/bin/pip install -e ".[dev,alfworld]"
.venv/bin/mcts-laya alfworld-data --out data/alfworld      # 約 1 分鐘
# 先跑一個 seed，確認速度與 warm start 的結果
.venv/bin/python scripts/run_ablation.py configs/ablation/alfworld.yaml --seeds 0
.venv/bin/python scripts/summarize_ablation.py runs/ablation/alfworld --reference T-teacher-only
```

預估成本（H100，32 個 actor；尚未實測，第一次執行請回報實際時間）：
- teacher 資料：本機 4 核測得每個程序 12 局約 45 秒；500 局 ÷ 31 個程序 ≈ 1–2 分鐘。
- 啟動：初始狀態是延遲載入的（第一次用到時才載入遊戲，約 1 秒），所以主程序不會先載入 300 多個評估遊戲；遊戲在 actor 裡載入。
- 每輪 self-play：128 局 × 平均約 30 步 × 16 次模擬；引擎約 1–2 分鐘，模型呼叫依 GPU 負載而定。
- 每次評估：驗證 60 局 + valid_unseen 134 局（warm start 與最後再加 valid_seen 140 局）。

## 6. 要回答的問題與預期

1. **warm start 有多好？** 規劃器知道物品位置，模型只能學到常見位置。預期 pick_and_place、look_at 較好，pick_two 最差。
2. **self-play 能不能學會搜尋？** 這是 ALFWorld 和 TextWorld 最大的不同：必須有系統地打開容器、記得看過哪裡。和 TextWorld 一樣以 T-teacher-only 為對照組做成對比較。
3. **搜尋的幫助有多大？** greedy vs PUCT16，各任務分開看。
4. 文獻參考（數字以原論文為準）：BUTLER（小模型、DAgger + beam search）在 valid_unseen 約 0.37；ReAct（PaLM-540B，few-shot）約 0.7。

## 7. 下一步

1. 使用者在 GPU 上跑 `configs/ablation/alfworld.yaml` 的 seed 0（control 與 T-teacher-only），回報 `summary.md` 與時間。
2. 依結果決定：如果 warm start 很差（例如 < 0.2），考慮改用「只用觀察到的資訊」的 teacher（需要另外寫搜尋腳本，約 2–3 天）或加上 DAgger；如果 self-play 有效，跑滿 3 個 seed。
3. 之後：和 LLM agent（ReAct）比較成功率、每任務成本與延遲。
