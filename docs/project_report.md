# Laya + MCTS：專案報告（架構、實驗與結果）

> 涵蓋範圍：專案開始到 Phase 1a 第二輪消融實驗（2026-10-07）
> 程式碼：分支 `claude/laya-mcts-project-planning-oj7zb4`
> 相關文件：[`candidate_problems.md`](../candidate_problems.md)（為什麼選這些任務）、[`ROADMAP.md`](../ROADMAP.md)（階段目標）、[`docs/phase0_report.md`](phase0_report.md)（Phase 0 細節）、[`docs/gpu_experiments.md`](gpu_experiments.md)（GPU 實驗操作步驟）

---

## 0. 一頁摘要

**做了什麼**：用 **Laya**（一個小型、非自回歸的「決策模型」）當 AlphaZero 裡的神經網路，搭配 **MCTS（蒙地卡羅樹搜尋）**，讓它在文字任務上透過 **self-play（自己玩、自己學）** 變強。

**主要結果**

| 問題 | 答案 | 證據 |
|---|---|---|
| 整個 AlphaZero 迴圈能不能運作？ | 能 | Phase 0：Countdown 上 Laya+搜尋 0.93，不用 Laya 的搜尋 0.30；代數從零開始可以學會 |
| self-play 在文字冒險遊戲（TextWorld）上有沒有幫助？ | **在需要探索的關卡上有，而且任務越難幫助越大** | 和「只用 teacher 資料反覆訓練」相比，同一個 seed 的成績：L2-goal +0.12、L3-goal +0.22，15 組比較全部為正 |
| 什麼情況下 self-play 沒有幫助？ | 模仿 teacher 就已經達到最優的關卡 | 每一步都有指示的 L1、L2，以及料理關卡 C1：warm start 後已接近理論最佳，self-play 與只用 teacher 資料的差距 ≤ 0.01 |
| self-play 為什麼有幫助？ | 主要是讓**價值網路**變得有用，搜尋因此變強 | 只用 teacher 資料時，搜尋只比不搜尋多 +0.04；加入 self-play 後多 +0.21 |
| 進階的 self-play 變體（只學有效率的局、更多模擬次數）有沒有更好？ | 沒有 | 和最簡單的版本差距 < 0.02，在雜訊範圍內 |
| 目前的瓶頸 | self-play 第 1–2 輪就趨於飽和；L1、L2、L2-goal、C1 都已經沒有比較空間 | L3-goal 約 70% 的進步發生在第 1 輪 |

**過程中最重要的工程結論**：需要 gate（只保留不退步的權重）、獨立的驗證集、平行對局（actor）才能讓 GPU 不閒置，以及對 TextWorld 本身的一個無限迴圈 bug 加上防護。

---

## 1. 背景

### 1.1 AlphaZero 與 Expert Iteration

AlphaZero（圍棋、西洋棋）的核心是兩個元件互相拉抬：

- **神經網路**看一個局面，輸出兩樣東西：
  - **policy（策略）**：每個合法動作的機率，代表「直覺上哪一步好」，也叫 **prior（先驗）**
  - **value（價值）**：這個局面最後會贏還是輸，介於 −1 到 1
- **MCTS（蒙地卡羅樹搜尋）**用網路的直覺當引導，往前試走很多步，得到比網路本身更好的決策。

訓練迴圈是：

1. 用「網路 + 搜尋」自己玩很多局（**self-play**）
2. 把搜尋得到的決策（比網路本身好）當成 policy 的訓練目標，把每局的最終結果當成 value 的訓練目標
3. 用這些目標訓練網路，網路變強 → 搜尋也變強 → 回到 1

這種「搜尋當老師、網路當學生」的做法又叫 **Expert Iteration**。

### 1.2 Laya 是什麼

Laya（Convai Innovations）是一個**非自回歸的決策模型**：它不像 LLM 一個字一個字生成，而是讀進「情境文字 + 問題 + 選項」，**一次前向計算**就輸出每個選項的機率。它的特點：

- 小（本專案用 `laya-multilingual`，644 MB，以多語 BERT 類編碼器為基礎）、快、可以在 CPU 上跑
- 支援多種題型。本專案用到兩種：
  - `choice`：多選一（拿來當 policy：選項就是候選動作）
  - `noul`：是 / 否（拿來當 value：「這個局面會成功嗎？」，V = 2·P(是) − 1）
- 每種題型有一個**溫度**參數，用來校準機率，讓「說 80% 的時候真的有 80% 會對」

所以 Laya 天然就能扮演 AlphaZero 網路的兩個角色，只是它看的是**文字**而不是棋盤。

### 1.3 為什麼選這些任務

棋類遊戲的狀態不適合用文字表達。我們需要**狀態本來就是文字、可以模擬、可以自動判斷對錯**的任務。完整評估在 [`candidate_problems.md`](../candidate_problems.md)，最後選擇：

- **Phase 0（驗證管線）**：Countdown（用四則運算湊出目標數字）、一元一次方程逐步化簡。便宜、可完美驗證。
- **Phase 1a（主線）**：**TextWorld**——微軟的文字冒險遊戲產生器，可以控制難度（房間數、物件數、任務步數）。
- **Phase 1b（之後）**：ALFWorld（建在 TextWorld 引擎上的家務任務 benchmark）。

---

## 2. 系統架構

### 2.1 整體迴圈

```
┌─ 一次實驗（mcts-laya run <設定檔>）──────────────────────────────────┐
│                                                                        │
│  0. 基準評估：不用 Laya 的搜尋（均勻先驗、隨機 rollout）              │
│  1. 初始評估：原始 Laya（還沒針對任務訓練）                          │
│  2. warm start：teacher（完美解題器）產生示範資料 → 訓練 Laya         │
│  3. 評估 → gate 記下參考分數                                          │
│  4. 重複 N 輪：                                                        │
│       a. self-play：Laya + MCTS 玩 64 局，記錄每一步的搜尋結果         │
│       b. 把每一步轉成訓練樣本（policy 目標、value 目標）放進 replay   │
│       c. 用 replay buffer 訓練 Laya，校準溫度                          │
│       d. gate：在驗證集上評分；退步超過容許值就還原成上一版權重       │
│       e. 在測試集上評估保留下來的權重                                  │
│  輸出：metrics.jsonl（每一步的紀錄）、summary.md、checkpoint          │
└────────────────────────────────────────────────────────────────────────┘
```

### 2.2 程式模組

所有程式在 `src/mcts_laya/`。各元件透過**註冊表（registry）**用名字組裝，換環境或換搜尋法只要改設定檔，不用改程式。

| 模組 | 作用 |
|---|---|
| `envs/` | 環境：Countdown、代數、TextWorld。統一介面：取得題目、合法動作、走一步、是否結束、結果、狀態的文字描述 |
| `evaluators/` | 「網路」：`LayaEvaluator`（Laya）、`uniform`（均勻先驗、價值 0）、`rollout`（隨機玩到底估價值） |
| `search/` | 搜尋：PUCT（AlphaZero 原版）、Gumbel（AlphaZero 的改良版）、greedy（不搜尋，直接選網路最高分的動作） |
| `teachers/` | 完美解題器，用來產生 warm start 示範資料 |
| `selfplay/` | 玩一局（`actor.py`）、訓練目標的篩選（`targets.py`）、平行對局（`parallel.py`） |
| `training/` | 訓練器與 replay buffer |
| `pipeline.py` | 上面的整個迴圈 |
| `viz/` | 視覺化介面（對局回放、搜尋樹、學習曲線） |
| `cli.py` | 指令列：`download`、`tw-games`、`run`、`viz`、`serve`、`bench` |

### 2.3 環境介面：函數式狀態

搜尋需要從同一個局面試很多不同的走法，所以**狀態必須可以複製、不能被修改**。每個環境都把狀態設計成不可變的資料：`step(狀態, 動作)` 回傳一個新狀態，舊狀態不變。

每個環境也支援**資料切分**（`split`）：`train`（訓練用）、`eval`（測試用）、`val`（驗證用，給 gate）。TextWorld 的三組遊戲由不同範圍的 seed 產生，彼此不重疊：

| 切分 | 遊戲數 | seed 範圍 |
|---|---|---|
| train | 150 | 0–149 |
| eval（測試） | 100 | 1,000,000–1,000,099 |
| val（驗證） | 40 | 2,000,000–2,000,039 |

### 2.4 Laya 如何當 policy / value 網路

每個局面會變成**兩個 Laya 問題**，共用同一段狀態文字。以下是 TextWorld L2-goal 的真實例子：

```
Goal: Please look and see that the Henderson's style chest is unlocked.
Recent actions: close toolbox.
You are carrying: a licorice strip and a nest of grubs.
Location: Studio. ... It looks like it's a Henderson's style chest. You make out a TextWorld box. ...
          There is a closed door leading west. There is an unguarded exit to the east.
Last result: You close the toolbox.
Steps left: 19.
```

- **policy 問題**（`choice`）：「Which command brings you closest to completing the goal?」選項 A–E 是合法指令：`drop licorice strip`、`drop nest of grubs`、`eat licorice strip`、`go east`、`open toolbox`。Laya 對每個選項給一個機率，就是 prior。
- **value 問題**（`noul`）：「Will the goal be completed within the remaining steps?」，P(是) 轉成 V = 2·P(是) − 1。

實作重點（`evaluators/laya_evaluator.py`）：
- Laya 內建的批次 API 要求同一批的問題相同，但搜尋樹的每個葉節點有不同的選項，所以我們自己組 token 序列、把很多葉節點**合成一批**送進模型。
- 依序列長度排序後分批，避免短的 value 問題被補齊成長的 policy 問題（Countdown 吞吐量因此從 1.6 提升到 4.1 rows/s）。
- 用快取避免重複計算同一個局面；模型更新後清空快取。

### 2.5 搜尋：PUCT

PUCT 是 AlphaZero 使用的 MCTS 變體。每次「模擬」從根節點往下，在每一層選分數最高的動作：

```
分數(a) = Q(a)  +  c_puct · P(a) · √N(父) / (1 + N(a))
          ↑ 經驗上這步多好     ↑ 先驗高、試得少的動作會被鼓勵
```

- `Q(a)`：走這步之後所有模擬結果的平均價值
- `P(a)`：Laya 給的先驗機率
- `N`：被拜訪的次數；`c_puct = 1.5` 控制「探索」和「利用」的比重

走到還沒展開的葉節點時，用 Laya 估它的 prior 和 value，再把 value 往上回傳、更新路徑上每個節點的 Q。模擬完後，**根節點各動作的拜訪次數分佈**就是搜尋的決策，也是 policy 的訓練目標。

本專案用到的設定：
- **模擬次數 16**（評估一律用 `puct16`），**批次 8**：一次選 8 個葉節點一起送進 Laya。為了避免 8 次都選到同一條路，用 **virtual loss**：選過的路徑暫時當成輸，讓下一次選別條。
- **Dirichlet 雜訊**（α = 0.3，占 25%）：只在 self-play 時加在根節點的先驗上，強迫探索網路不看好的動作。
- 拜訪次數平手時依序比 Q 值、先驗（Phase 0 發現：模擬次數等於批次大小時拜訪數常常平手）。

另外實作了 **Gumbel AlphaZero**（以抽樣加循序淘汰取代 PUCT 的根節點選擇，模擬次數少時理論上較好）。Phase 0 的結果是價值網路弱時 PUCT 比較穩定，所以後續都用 PUCT。

### 2.6 self-play 與訓練目標

一局 self-play 的每一步產生一個訓練樣本：

| 目標 | 內容 | 細節 |
|---|---|---|
| policy | 搜尋的拜訪次數分佈 | 先做 **銳化**：取 1/0.25 次方再正規化（`policy_target_temperature: 0.25`）。16 次模擬的分佈很平，不銳化會把 warm start 學到的清楚偏好稀釋掉 |
| value | 這局的最終結果 | 成功：`0.5 + 0.5 · 0.9^步數`（轉成 −1~1）；失敗：0 |
| 篩選 | 失敗局只訓練 value，不訓練 policy | 不要模仿失敗的走法（`failed_policy_weight: 0`） |

**reward 公式的設計**：`0.5 + 0.5 · 0.9^步數`。成功至少拿 0.5，所以「解出來」一定明顯好過「沒解出來」；步數越少分數越高，所以也鼓勵效率。（Phase 0 一開始只用 `0.9^步數`，結果晚解出的分數接近 0，價值網路寧可拖延，見 §6。）

replay buffer 保留**全部** teacher 資料（`teacher_mix: 1.0`）加上所有 self-play 資料；每一輪用整個 buffer 訓練 1 個 epoch。

### 2.7 訓練器

`training/trainer.py`：
- **損失**：soft cross-entropy——讓 Laya 對選項的機率分佈接近目標分佈（policy 是拜訪分佈；value 是 [1−p, p]）。
- **選項順序擴增**：訓練時隨機打亂選項順序，避免模型記住「答案通常是 A」。
- **省記憶體**：凍結 token embedding（multilingual 模型的 25.6 萬字詞表占了約 60% 參數），並用 gradient checkpointing。
- **溫度校準**：每輪留 10%（最多 400 筆）資料不訓練，用它找出讓機率最準的溫度。
- 在 CUDA 上用混合精度（Ampere 以上 bf16；T4 / V100 用 fp16 加 loss scaling）。

### 2.8 Gate 與評估設計

**Gate**（借自 AlphaGo Zero）：每輪訓練完，用驗證集打分，若 PUCT16 reward 比目前最佳低超過 0.02，就**還原成上一版權重**。原因見 §4.2：沒有 gate 時 self-play 會讓模型退步。

**評估設計演進**（重要，因為它影響結果能不能信）：

| 版本 | 做法 | 問題 |
|---|---|---|
| 第一輪 | gate 和報告用同一組 40 局測試遊戲 | 選模型時看過測試題，報告的分數偏高；最後報告的「final」可能是被 gate 退回的那一版 |
| **第二輪（v2，目前）** | gate 用 40 局**驗證**遊戲決定保留與否；保留下來的權重才在 100 局**測試**遊戲上評估；被退回的那輪直接沿用上一版的測試結果 | — |

### 2.9 平行對局（actor 架構）

最初一次只玩一局，GPU 使用率只有 0–1%：TextWorld 每展開一個節點都要重播整段歷史（約 50 ms，只用一個 CPU 核心），而 GPU 處理一批只要幾毫秒，大部分時間都在等 CPU。

改成 AlphaZero 的做法（`selfplay/parallel.py`）：

```
actor 1..N（各自一個程序）：環境 + 搜尋樹 + 把局面轉成 token
        │  葉節點請求
        ▼
主程序（持有 GPU 上的模型）：收集所有 actor 的請求 → 合併成一批 → 前向計算 → 回傳
```

- actor 數量預設為「CPU 核心數 − 1」（最多 32），只在模型放在 GPU 上時啟用。
- token 化在 actor 裡做：每個局面約 1.4 ms，若由主程序做，主程序會變成新的瓶頸。
- 每一局有自己的亂數種子，結果不受「哪個 actor 玩、何時玩」影響（只有批次組成可能造成極小的浮點差異）。
- **安全機制**：每個 actor 的記憶體上限 16 GB、每局時間上限 30 分鐘；超過或出錯時那一局記為失敗（self-play 捨棄、評估算輸），換新的 actor，實驗繼續。

實測（使用者的共用 H100 機器，同時跑 5 個實驗）：基準評估快 4.6 倍，整個實驗從 72 分鐘降到 41 分鐘。

### 2.10 其他工具

| 工具 | 用途 |
|---|---|
| `mcts-laya viz` / `serve` | HTML 視覺化：對局回放（每一步的 prior、搜尋分佈、Q、價值）、搜尋樹、同題比較、學習曲線；TextWorld 另外畫出地圖 |
| `scripts/run_ablation.py` | 依設定檔批次跑「關卡 × 變體 × seed」，可中斷續跑（有 `done.json` 的會跳過），`--jobs N` 平行 |
| `scripts/summarize_ablation.py` | 彙整成表格：各變體跨 seed 的平均 ± 標準差、和參考變體的**成對差距** |
| 進度顯示 | 終端機上是 tqdm 進度條；寫進 log 檔時每 30 秒一行文字 |
| `--cpus N` | 限制使用的 CPU 核心數（共用機器時用） |

---

## 3. 任務

### 3.1 Phase 0：Countdown 與代數

- **Countdown**：給幾個數字和一個目標，每一步選兩個數做四則運算，最後要湊出目標。reward = 是否成功。teacher 是窮舉搜尋。
- **一元一次方程**：例如 `5 = (x+3)/3 + (x+3)/2`，每一步選一個代數改寫（展開、移項、通分……），化成 `x = 3` 即成功，用 SymPy 驗證每一步正確。reward = `0.5 + 0.5·0.9^步數`。teacher 是廣度優先搜尋，找最短步數。

### 3.2 Phase 1a：TextWorld

**關卡**（由 `mcts-laya tw-games` 產生）：

| 關卡 | 房間 | 物件 | 任務步數 | 目標怎麼寫 | 步數上限 |
|---|---|---|---|---|---|
| L1 | 3 | 6 | 3 | 每一步都寫出來 | 10 |
| L2 | 5 | 10 | 5 | 每一步都寫出來 | 15 |
| L2-goal | 5 | 10 | 5 | **只寫最終目標**（要自己探索） | 20 |
| L3-goal | 8 | 15 | 8 | 只寫最終目標 | 30 |
| C1 | 1 種食材、1 個房間 | | | 料理遊戲：先讀食譜再照做 | 15 |
| C2 | 2 種食材、6 個房間 | | | 料理遊戲 | 30 |

「只寫最終目標」的意思是：例如目標只說「確認某個箱子已解鎖」，agent 要自己找到鑰匙在哪、怎麼過去。這需要探索和記憶，是 self-play 能發揮的地方。

**狀態文字**（見 §2.4 的例子）包含：目標（去掉 TextWorld 隨機產生的問候語）、最近 6 個動作、走過的房間、讀過的重要內容（如食譜）、物品欄、所在位置描述、上一步的結果、剩餘步數。`look`、`inventory`、`examine` 這類不改變世界的指令會被濾掉（料理關卡保留 `examine cookbook`）。

**teacher**：TextWorld 本身會追蹤任務進度，提供每個狀態的最佳剩餘指令（`policy_commands`）。teacher 以 30% 機率隨機走一步（`explore: 0.3`），讓示範資料也涵蓋偏離最佳路線的局面。料理關卡的 teacher 會先讀食譜，因為 TextWorld 的最佳路線會跳過這步（它本來就知道食譜），但 agent 不讀就不知道要做什麼。

**實作重點**：
- **狀態 = 遊戲 + 動作歷史**。要得到某個狀態的觀察，就重置遊戲再重播歷史（重置約 5 ms、每步約 7 ms），因為 TextWorld 的 `copy()` 要 52 ms，而直接存還原模擬器狀態會讓 TextWorld 的任務追蹤不同步。
- 每個遊戲保留最多 4 個引擎，搜尋在分支間切換時，挑一個已經停在「請求歷史的前綴」上的引擎接著走；觀察結果用 (遊戲, 歷史) 做快取。引擎時間因此減少約 30%，結果完全不變（有測試驗證）。

---

## 4. 實驗與結果

**怎麼讀數字**
- 主要指標是 **PUCT16 reward**：用 Laya + 16 次模擬的 PUCT 玩評估遊戲的平均 reward。成功一局的 reward 是 `0.5 + 0.5·0.9^步數`（5 步約 0.795、7 步約 0.739、8 步約 0.715），失敗是 0。
- 同時記錄三個輔助指標：
  - **成功率**：解出的局數比例。
  - **平均步數**：所有測試局的平均步數。失敗的局也算進來，而 TextWorld 的 L 關卡沒有「輸」的條件，失敗一定是走到步數上限（L1 10 步、L2 15、L2-goal 20、L3-goal 30），所以失敗越多，這個數字越大。
  - **成功局平均步數**：只算解出的局，代表「解得多有效率」。實驗紀錄從 2026-10-08 起直接記錄這一項（`success_moves`）；之前的結果在 L 關卡上可以由前兩項精確換算：（平均步數 −（1 − 成功率）× 步數上限）÷ 成功率；`scripts/summarize_ablation.py` 會自動換算，彙整表的 `success moves` 欄已包含這些結果。
- **greedy**：不搜尋、直接選 Laya 最高機率的動作，代表「網路本身」的能力。
- **基準**：`uniform-puct16`（同樣的搜尋，但先驗均勻、價值 0，等於沒有 Laya）；`rollout-puct16`（用隨機玩到底估價值的傳統 MCTS）。
- **雜訊**：40 局的評估，reward 的不確定性約 ±0.03–0.05；100 局約 ±0.02–0.03。差距比這小就不能下結論。
- **±** 是跨 seed 的標準差；**成對差距**是每個 seed 和同 seed 的參考組相減再平均（warm start 依 seed 而異，相減可以抵消這部分差異）。

### 4.1 Phase 0：管線驗證（CPU，2026-10-03）

完整內容見 [`docs/phase0_report.md`](phase0_report.md)。

| 實驗 | 結果（PUCT16） | 不用 Laya 的搜尋 |
|---|---|---|
| Countdown，真 Laya，teacher 150 題 + 1 輪 self-play | 0.17 → 0.83 → **0.93** | 0.30 |
| 代數，真 Laya，teacher 150 題 | 0.35 → 0.776（理論最優約 0.777） | 0.727 |
| 代數，**微型模型，從零開始**（無 teacher），8 輪 | greedy 0.00 → 0.61，PUCT16 0.07 → 0.74 | 0.730 |

結論：迴圈可以運作；預訓練的 Laya 比隨機初始化的微型模型好學得多；不用 teacher、純 self-play 也能學會（真正的 AlphaZero 式學習）。

### 4.2 Phase 1a 前期：CPU 縮小版（評估 20–30 局）

| 關卡 | warm start（PUCT16 reward / 成功率） | self-play 後 |
|---|---|---|
| L1 | 0.837 / 0.97 | **0.866 / 1.00**（最優） |
| L2 | 0.761 / 1.00 | 0.71–0.76（持平） |
| L2-goal，無 gate | 成功率 0.95 | **退步到 0.65** |
| L2-goal，有 gate | 0.754 | 3 輪都被退回（0.611 / 0.534 / 0.694） |

**問題分析**：self-play 的成功局平均要 8–10 步，最佳約 5 步。模型在模仿自己繞遠路的走法，而且 16 次模擬太淺，搜尋給的目標不比網路本身好。這引出 §4.3 的消融實驗。

### 4.3 GPU 消融第一輪：哪種 self-play 目標有用？

設計了 7 個變體，在 L2-goal 上各跑 3 個 seed：

| 變體 | 做法 | 要檢驗的假設 |
|---|---|---|
| `control` | 成功局的搜尋結果當 policy 目標（預設做法） | 對照組 |
| `A-efficient` | 只學「有效率」的成功局：步數 ≤ 1.3 × 自己在該題的最佳紀錄，且 reward 在本輪前 50% | 退步來自模仿繞遠路 |
| `B-value-only` | self-play 只訓練 value，policy 只從 teacher 學 | policy 被模糊的目標稀釋 |
| `C-sims48` | self-play 用 48 次模擬（評估仍是 16） | 搜尋太淺 |
| `AC`、`BC` | 上面兩兩組合 | 修正可以疊加 |
| `D-dagger` | teacher 為 self-play 走過的每個狀態重新標註最佳動作與價值 | 模仿學習的上限參考 |

結果（使用者的 H100，PUCT16 reward）：

| 變體 | final | 相對 warm start |
|---|---|---|
| A-efficient | 0.702 ± 0.024 | +0.111 |
| C-sims48 | 0.681 ± 0.027 | +0.059 |
| control | 0.677 ± 0.053 | +0.079 |
| AC | 0.653 ± 0.010 | +0.025 |
| BC | 0.648 ± 0.105 | +0.035 |
| D-dagger | 0.638 ± 0.025 | +0.043 |
| B-value-only | 不可用（一個 seed 沒跑完） | |

**這一輪的結論很有限**，因為發現了幾個問題：
1. gate 和報告用同一組測試遊戲（§2.8），分數偏高。
2. 部分實驗中途重啟，舊紀錄和新紀錄混在同一個檔案（gate 欄位出現 8+10 這種分母）。
3. 有一個 seed 沒跑完卻被算進平均。
4. **缺少關鍵對照組**：每一輪都用整個 replay buffer（含全部 teacher 資料）重新訓練，到第 8 輪 teacher 資料已經被學了約 10 次。所以「比 warm start 進步」可能只是因為 teacher 資料學得比較多，而不是 self-play 的功勞。

這些問題全部修正後，進行第二輪。

### 4.4 GPU 消融第二輪（v2）：self-play 到底有沒有用？

**改了什麼**
- 評估改用 v2 設計（gate 用驗證集、報告用 100 局測試集，§2.8）。
- 新增對照組 **`T-teacher-only`**：迴圈完全一樣，但 self-play 的資料**不加入** replay buffer，每輪只用 teacher 資料重新訓練。若 control 勝過它，進步才是 self-play 的功勞。
- 彙整腳本只採用每個實驗最後一次的紀錄，沒跑完的另外列出，並計算和 T-teacher-only 的成對差距。

**共同設定**：teacher 150 題、warm start 2 個 epoch、8 輪 self-play、每輪 64 局、PUCT 16 次模擬（批次 8）、policy 目標銳化 0.25、gate 容許 0.02、3 個 seed。

#### 結果總表（PUCT16，100 局測試，最終保留的權重）

| 關卡 | 變體 | final reward | 成功率 | 平均步數 | 成功局平均步數 | 相對 warm start | **相對 teacher-only（成對）** | greedy |
|---|---|---|---|---|---|---|---|---|
| L2-goal | T-teacher-only | 0.613 ± 0.033 | 0.79 ± 0.05 | 8.8 ± 0.5 | 5.8 ± 0.2 | +0.015 | — | 0.579 |
| L2-goal | control | 0.736 ± 0.022 | 0.94 ± 0.03 | 6.5 ± 0.5 | 5.6 ± 0.2 | +0.140 | **+0.123 ± 0.049** | 0.590 |
| L2-goal | A-efficient | 0.727 ± 0.035 | 0.94 ± 0.04 | 6.9 ± 0.9 | 6.0 ± 0.4 | +0.120 | **+0.115 ± 0.043** | 0.612 |
| L3-goal | T-teacher-only | 0.413 ± 0.060 | 0.59 ± 0.08 | 18.3 ± 2.1 | 10.2 ± 1.0 | +0.043 | — | 0.371 |
| L3-goal | control | 0.640 ± 0.004 | 0.89 ± 0.01 | 11.0 ± 0.3 | 8.6 ± 0.5 | +0.219 | **+0.227 ± 0.061** | 0.428 |
| L3-goal | A-efficient | 0.643 ± 0.026 | 0.89 ± 0.04 | 11.0 ± 0.9 | 8.7 ± 0.6 | +0.249 | **+0.231 ± 0.063** | 0.445 |
| L3-goal | C-sims48 | 0.624 ± 0.036 | 0.88 ± 0.04 | 11.6 ± 1.3 | 9.1 ± 0.5 | +0.223 | **+0.211 ± 0.025** | 0.465 |
| C1 | 全部變體（4 個） | 0.804 | 1.00 | 4.7 | 4.7 | 0.000 | 0.000 | 0.804 |

平均步數含失敗局（L2-goal 以 20 步、L3-goal 以 30 步計）；成功局平均步數只算解出的局。任務本身的最短步數：L2-goal 5 步、L3-goal 8 步。原始數據：`runs/ablation/selfplay_textworld_v2/summary.md`。

#### 發現 1：self-play 有幫助，而且任務越難幫助越大

每個 seed 都和同 seed 的 teacher-only 比較：

| 關卡 | control | A-efficient | C-sims48 |
|---|---|---|---|
| L2-goal | +0.142、+0.067、+0.161 | +0.152、+0.068、+0.123 | — |
| L3-goal | +0.162、+0.235、+0.283 | +0.178、+0.213、+0.300 | +0.187、+0.209、+0.237 |

15 組全部為正。L3 平均 +0.22（t ≈ 6），L2 平均 +0.12。

只用 teacher 資料反覆訓練幾乎沒有幫助（L2 +0.015、L3 +0.04），甚至會過度擬合：L3 的 teacher-only 在訓練資料上的 policy 準確率到 0.98，但 seed 2 的測試成績從 0.474 掉到 0.361。

#### 發現 2：進階變體沒有比較好

L3 上 control、A-efficient、C-sims48 的差距都小於 0.02，在雜訊範圍內。C-sims48 的 self-play 時間是 1.5–2 倍，卻沒有更好的成績。**結論：用最簡單的 control 即可。**

#### 發現 3：self-play 的作用主要在價值網路

| L3-goal，第 8 輪 | value 損失 | 保留資料上的 value 誤差（Brier） | **搜尋帶來的提升**（PUCT16 − greedy） |
|---|---|---|---|
| T-teacher-only | 0.626–0.631，不再下降 | ≈ 0.000 | **+0.04** |
| control | 0.495–0.503 | 0.014–0.026 | **+0.21** |

解讀：
- teacher 每一局都成功，所以 teacher 資料裡每個局面的價值標籤都是「會成功」。value 網路很容易學（誤差 ≈ 0），但它分不出好局面和壞局面，對搜尋沒有用處，搜尋只比不搜尋好 0.04。
- self-play 帶進了**真實的結果**——包括失敗、繞路——而且正好是搜尋實際會走到的局面。value 網路學會分辨好壞，搜尋就變得有用：比不搜尋好 0.21。
- L2 也一樣：搜尋帶來的提升，teacher-only 是 +0.03，control 是 +0.15。
- 這也解釋了第一輪 D-dagger 為什麼沒有變好：teacher 給的價值標籤假設「之後都走最佳路線」，太樂觀，同樣讓價值網路失去分辨能力。

#### 成功率與步數

reward 同時反映「有沒有解出」和「用了幾步」。把結果總表的成功率與步數欄拆開來看：

- self-play 的進步**主要來自成功率**：L3-goal 從 0.59 到 0.89、L2-goal 從 0.79 到 0.94。
- **解出的局也變得更有效率**：L3-goal 從 10.2 步降到 8.6 步，已經接近任務本身的 8 步；L2-goal 5.8 → 5.6 步，任務是 5 步。
- 「平均步數」從 18.3 降到 11.0，大部分是因為失敗的局變少（每一局失敗都以 30 步計），不代表每一局都少走了 7 步。
- 進階變體在步數上也沒有優勢：L3-goal 的 A-efficient（只學有效率的成功局）成功局 8.7 步，和 control 的 8.6 步相同；C-sims48 是 9.1 步，反而略多。L2-goal 的 A-efficient 6.0 步也沒有比 control 的 5.6 步少。

#### 發現 4：self-play 很快就飽和

L3 control 第 1 輪後平均 0.580，最後 0.640，約 70% 的進步發生在第 1 輪。之後 gate 每 8 輪只接受 2–5 輪，而且很多決定落在驗證集（40 局）的雜訊範圍內。L3 目前約 89% 的局成功，解出的局平均 8.6 步，已經接近任務的 8 步；如果每局都以這個步數解出，reward 約 0.70，所以剩下約 0.06 的空間**主要在成功率**（還有約 11% 的局沒解出），而不是效率。L2 已經到上限（約 0.74）。

#### 發現 5：C1 太簡單，無法比較

warm start 之後，所有變體、所有 seed 都是 100% 成功、平均 4.73 步，reward 0.804 = 0.5 + 0.5·0.9^4.73，也就是完全照著 teacher 的路線走，greedy 和搜尋的結果一模一樣。料理關卡 C1 的步驟太短（讀食譜、照做），模仿就能完全解決，沒有留給 self-play 的空間。

### 4.5 各階段對照：未微調的 Laya、greedy、warm start 與 self-play

每個實驗開始時都會先評估**未經任何微調的原始 Laya**（`initial`），每次評估也都同時跑 greedy（不搜尋）和 PUCT16（搜尋），所以從既有的紀錄就能整理出「從原始模型到 self-play 之後」每一步的效果。

**GPU 第二輪（100 局測試）**：每一列是該關卡所有跑完的實驗的平均（L2-goal 9 個、L3-goal 12 個、C1 11 個）。原始 Laya 在每個實驗都相同，因為模型和測試遊戲都一樣。括號內是成功率。

| 方法 | L2-goal PUCT16 | L2-goal greedy | L3-goal PUCT16 | L3-goal greedy | C1 PUCT16 | C1 greedy |
|---|---|---|---|---|---|---|
| 均勻先驗搜尋（不用 Laya） | 0.032（0.04） | — | 0.048（0.06） | — | 0.027（0.00） | — |
| 隨機 rollout 搜尋（不用 Laya） | 0.233（0.34） | — | 0.094（0.13） | — | 0.277（0.32） | — |
| **原始 Laya（未微調）** | **0.128（0.19）** | **0.061（0.09）** | **0.052（0.07）** | **0.031（0.04）** | **0.140（0.15）** | **0.003（0.00）** |
| warm start 後 | 0.601 | 0.501 | 0.396 | 0.303 | 0.804（1.00） | 0.804（1.00） |
| self-play 後（control） | 0.736 | 0.590 | 0.640 | 0.428 | 0.804 | 0.804 |
| 只用 teacher 資料重新訓練（T-teacher-only） | 0.613 | 0.579 | 0.413 | 0.371 | 0.804 | 0.804 |

**CPU 縮小版（Phase 1a 前期，評估 20–30 局、3 輪 self-play）**：

| 關卡 | 均勻先驗搜尋 | 隨機 rollout | **原始 Laya PUCT16 / greedy** | warm start PUCT16 / greedy | self-play 後 PUCT16 / greedy |
|---|---|---|---|---|---|
| L1（30 局） | 0.245 | 0.595 | **0.375 / 0.193** | 0.837 / 0.779 | 0.866 / 0.865 |
| L2（30 局） | 0.028 | 0.173 | **0.175 / 0.069** | 0.761 / 0.694 | 0.708 / 0.662 |
| L2-goal（20 局，有 gate） | 0.041 | 0.286 | **0.146 / 0.072** | 0.754 / 0.661 | 3 輪都被 gate 退回，保留 warm start |

（原始數據：`docs/results/phase1a/`。這些是縮小版，評估局數少，雜訊約 ±0.05–0.08。）

**解讀**
- **原始 Laya 幾乎不能直接用**：所有關卡的 greedy 成功率都在 10% 以下。L2-goal 上，原始 Laya 加搜尋（0.128）勝過均勻先驗（0.032），代表它對情境有一點判讀能力，但仍不如隨機 rollout（0.233）；L3 上則和沒有模型一樣（0.052 vs 0.048）。Laya 是通用的決策模型，沒看過 TextWorld 的指令和任務格式，需要針對任務微調。
- **teacher 微調是最大的一步**：greedy 在 L2-goal 從 0.06 到 0.50，在 L3 從 0.03 到 0.30。
- **self-play 主要提高「搜尋之後」的成績**：從 warm start 到 self-play 後，PUCT16 在 L3 增加 0.24，greedy 只增加 0.13；和 §4.4 發現 3 一致，self-play 主要改善的是價值網路，而價值網路只有在搜尋時才派上用場。
- L1、L2 的完整規模結果見 §4.6。

### 4.6 L1、L2 完整規模：模仿已經達到最優時，self-play 沒有空間

用和第二輪相同的評估設計（`configs/ablation/phase1a_full.yaml`：100 局測試、40 局驗證、gate），在每一步都有指示的 L1、L2 上跑 T-teacher-only 與 control，各 3 個 seed。

**理論最佳**：L1 的任務是 3 步，全部用 3 步解出時 reward = 0.5 + 0.5·0.9³ ≈ **0.864**；L2 是 5 步，約 **0.795**。

| 方法 | L1 PUCT16 | L1 greedy | L2 PUCT16 | L2 greedy |
|---|---|---|---|---|
| 均勻先驗搜尋（不用 Laya） | 0.276（0.34） | — | 0.032（0.04） | — |
| 隨機 rollout 搜尋（不用 Laya） | 0.564（0.70） | — | 0.156（0.21） | — |
| **原始 Laya（未微調）** | **0.446（0.59）** | **0.254（0.31）** | **0.117（0.18）** | **0.042（0.06）** |
| warm start 後（6 個實驗平均） | 0.859 | 0.837 | 0.772 | 0.733 |
| self-play 後（control） | 0.848 ± 0.014 | 0.829 | 0.780 ± 0.004 | 0.766 |
| 只用 teacher 資料重新訓練（T-teacher-only） | 0.857 ± 0.006 | 0.837 | 0.776 ± 0.002 | 0.753 |
| **control − T-teacher-only（成對）** | **−0.009 ± 0.008** | | **+0.004 ± 0.006** | |

（括號內是成功率。）

最終保留的權重（PUCT16，100 局測試，3 個 seed）：

| 關卡 | 變體 | final reward | 成功率 | 平均步數 | 成功局平均步數 | 任務步數 | gate 接受 |
|---|---|---|---|---|---|---|---|
| L1 | T-teacher-only | 0.857 ± 0.006 | 0.99 ± 0.01 | 3.1 ± 0.1 | 3.1 ± 0.0 | 3 | 8/8、8/8、7/8 |
| L1 | control | 0.848 ± 0.014 | 0.98 ± 0.02 | 3.2 ± 0.2 | 3.1 ± 0.1 | 3 | 8/8、8/8、8/8 |
| L2 | T-teacher-only | 0.776 ± 0.002 | 0.98 ± 0.01 | 5.5 ± 0.1 | 5.3 ± 0.1 | 5 | 8/8、7/8、8/8 |
| L2 | control | 0.780 ± 0.004 | 0.99 ± 0.01 | 5.4 ± 0.0 | 5.2 ± 0.0 | 5 | 8/8、6/8、6/8 |

（原始數據：`runs/ablation/phase1a_full/summary.md`。平均步數含失敗局，L1 以 10 步、L2 以 15 步計。）

**解讀**
- **warm start 就已經接近最優**：L1 在 warm start 後平均 0.859，理論最佳 0.864，成功率 98–100%、平均約 3.1 步；L2 是 0.772，最佳約 0.795，成功率 97–99%。self-play 能改進的空間只剩 0.005（L1）和 0.02（L2），小於評估的雜訊。
- **self-play 和只用 teacher 資料沒有差別**（L1 −0.009、L2 +0.004，都在雜訊範圍內）。成功率與步數也一樣（上表）：L1 兩者的成功率 0.98–0.99、成功局平均 3.1 步，正好是任務的 3 步；L2 成功率 0.98–0.99、成功局平均 5.2–5.3 步，任務是 5 步。離理論最佳剩下的差距，L1 幾乎都來自 1–2% 沒解出的局；L2 則是沒解出的局和成功局多走的 0.2–0.3 步各占一部分。這兩個關卡每一步都有指示，照著指示做就是最佳解，模仿 teacher 已經足夠，不需要探索。
- **L1 的 control 有兩個 seed 最後略低**（0.840、0.839，warm start 0.863、0.862），但 gate 每一輪都接受。原因是驗證集上的成績一直停在最高值附近（0.860–0.864），落在容許範圍（0.02）內，而測試集上 0.02 左右的下降屬於雜訊等級，gate 不會、也不應該對這種差距做反應。
- **搜尋本身的幫助也很小**：PUCT16 只比 greedy 高 0.01–0.02，因為 policy 已經幾乎每一步都選對。
- **原始 Laya 在 L1 有一點基礎能力**：搜尋 0.446 勝過均勻先驗 0.276，但不如隨機 rollout 0.564；L1 只有 3 個房間，隨機玩到底就常常能成功。

**跨關卡的整體圖像**（self-play 相對 T-teacher-only 的成對差距）：

| 關卡 | 目標寫法 | warm start 後離最佳還有多遠 | self-play 的幫助 |
|---|---|---|---|
| L1 | 每一步都寫 | 約 0.005 | −0.009（無） |
| L2 | 每一步都寫 | 約 0.02 | +0.004（無） |
| C1 | 料理：讀食譜照做 | 0（已最優） | 0.000（無） |
| L2-goal | 只寫最終目標 | 約 0.14（warm start 0.60，最好的變體達 0.74） | **+0.12** |
| L3-goal | 只寫最終目標 | 約 0.30（warm start 0.40，理想約 0.70） | **+0.22** |

self-play 的幫助大小和「模仿 teacher 之後還剩多少空間」一致：只有需要自己探索的關卡，模仿留下了空間，self-play 才有東西可以學。

---

## 5. 結論

1. **Laya + MCTS 的 AlphaZero 式迴圈在文字任務上有效。** 在需要探索的 TextWorld 關卡上，self-play 帶來明確且可重現的進步，任務越難進步越大；在模仿 teacher 就能達到最優的關卡（L1、L2、C1）上則沒有差別，因為已經沒有改進空間（§4.6）。
2. **進步的來源主要是價值網路。** teacher 只能教「最佳路線長什麼樣子」，教不出「這個局面有多糟」；self-play 補上了這一塊，讓搜尋真正發揮作用。這呼應 AlphaZero 的設計：搜尋的品質取決於價值估計。
3. **簡單的設定就夠了。** 目前沒有任何進階變體勝過「成功局的搜尋分佈 + 銳化 + gate」。
4. **目前的限制**：self-play 在 1–2 輪後就飽和；L1、L2、L2-goal、C1 都已經沒有比較空間，目前只有 L3-goal 還能分辨不同做法；gate 的驗證集只有 40 局，雜訊偏大。

---

## 6. 過程中解決的重要問題

| 問題 | 症狀 | 原因 | 解法 |
|---|---|---|---|
| reward 讓晚解出的分數接近 0 | 價值網路傾向拖延 | `0.9^步數` 在步數多時趨近 0，比失敗好不了多少 | `success_floor`：成功至少 0.5 |
| self-play 讓模型退步 | L2-goal 成功率 0.95 → 0.65 | 模仿繞遠路的成功局、16 次模擬的目標太平 | gate + policy 目標銳化（0.25） |
| GPU 幾乎閒置 | 使用率 0–1% | 一次一局，CPU 上的 TextWorld 重播是瓶頸 | 平行 actor 架構（§2.9）、引擎池與觀察快取 |
| 評估偏高 | 報告的分數可能來自被退回的權重 | gate 和報告用同一組測試題 | 驗證集 / 測試集分開（§2.8） |
| 無法判斷是不是 self-play 的功勞 | 「比 warm start 進步」 | teacher 資料每輪都被重新訓練 | 新增 T-teacher-only 對照組 |
| 重啟的實驗資料混雜 | gate 欄位出現 8+10 | `metrics.jsonl` 用附加模式寫入 | 新的一次實驗會先把舊檔改名；彙整只取最後一次 |
| **TextWorld 本身的無限迴圈** | 單一 actor 記憶體用到 300 GB、實驗卡住數小時 | TextWorld 每走一步會重新規劃任務路線；在某些偏離路線的狀態下，規劃演算法移除一個步驟時又把它的反向步驟（開 / 關、來 / 回）推回去，永遠不會結束 | 規劃超過 100 步（正常最多 23 步）時，放棄該任務接下來的路線提示（不影響勝負判定、分數與合法指令）；再加上 actor 記憶體上限與每局時間上限 |
| 遊戲名稱被當成雜訊刪掉 | 約 3% 的遊戲目標文字缺字 | 清理「TextWorld 問候語」的規則也刪掉了名字裡有 TextWorld 的物件 | 改用句型比對 |
| 套件檔案沒有上傳到 GitHub | 使用者安裝後出現 `No module named 'mcts_laya.models'` | `.gitignore` 的 `models/` 也排除了 `src/mcts_laya/models/` | 改成只排除根目錄的 `/models/` |

---

## 7. 下一步建議

依優先順序：

1. **測試「評估時多搜尋」的效果**：用已訓練好的模型，以 16 / 32 / 64 / 128 次模擬評估。如果 self-play 訓練出的價值網路讓「搜尋越多、成績越好」，就是 AlphaZero 式「搜尋可以擴展」的核心證據，也能判斷 L3 的飽和是網路的限制還是搜尋預算的限制。不需要重新訓練，只需要新增一個評估指令。
2. **確認機制**：在 L3 上跑 B-value-only（self-play 只訓練 value）。如果它和 control 一樣好，就確定 self-play 的貢獻來自價值網路。
3. **更難的任務**：C2（多食材、多房間的料理遊戲），或開始 Phase 1b 的 ALFWorld。
4. **讓 self-play 持續進步**：較大的驗證集（降低 gate 的雜訊）、每輪更多局、逐步降低 teacher 資料的比重。

---

## 附錄 A：如何重現

```bash
# 環境
scripts/setup_env.sh --download                       # 建 venv、安裝、下載 laya-multilingual
.venv/bin/pip install -e ".[dev,textworld]"           # TextWorld
.venv/bin/python -m pytest                            # 全部測試

# 產生 TextWorld 遊戲池
.venv/bin/mcts-laya tw-games --level L2-goal L3-goal C1 --train 150 --eval 100 --val 40 --out data/textworld

# 單一實驗
.venv/bin/mcts-laya run configs/textworld/l3-goal.yaml --set model.device=cuda

# 第二輪消融（GPU）與彙整
.venv/bin/python scripts/run_ablation.py configs/ablation/selfplay_textworld.yaml --only L2-goal L3-goal \
    --variants T-teacher-only control --seeds 0 1 2 --jobs 2
.venv/bin/python scripts/summarize_ablation.py runs/ablation/selfplay_textworld_v2 --reference T-teacher-only

# L1、L2 的完整規模版本（§4.6）
.venv/bin/mcts-laya tw-games --level L1 L2 --train 150 --eval 100 --val 40 --out data/textworld
.venv/bin/python scripts/run_ablation.py configs/ablation/phase1a_full.yaml --jobs 2
.venv/bin/python scripts/summarize_ablation.py runs/ablation/phase1a_full --reference T-teacher-only

# 視覺化某個實驗
.venv/bin/mcts-laya viz --run runs/ablation/selfplay_textworld_v2/L3-goal/control-s0 --problems 4
```

詳細的 GPU 操作步驟（硬體需求、記憶體不足時的調整、平行設定）見 [`docs/gpu_experiments.md`](gpu_experiments.md)。

## 附錄 B：名詞對照

| 名詞 | 意思 |
|---|---|
| policy / prior | 網路對每個合法動作給的機率，代表直覺上哪一步好 |
| value | 網路對局面結果的估計（−1 到 1） |
| MCTS / PUCT | 蒙地卡羅樹搜尋；PUCT 是 AlphaZero 用的版本 |
| 模擬（simulation） | 搜尋從根節點往下走到一個葉節點並回傳價值的一次過程 |
| greedy | 不搜尋，直接選網路機率最高的動作 |
| self-play | 用目前的網路 + 搜尋自己玩，產生訓練資料 |
| teacher / warm start | 用完美解題器的示範資料先訓練網路，讓 self-play 有個起點 |
| replay buffer | 存放訓練樣本的資料池 |
| gate | 每輪訓練後打分，退步太多就還原上一版權重 |
| seed | 亂數種子；不同 seed 代表獨立重複一次實驗 |
| 成對差距 | 同一個 seed 下兩個變體的差，再跨 seed 平均 |
| actor | 平行對局架構中負責玩遊戲的程序 |
| reward | `0.5 + 0.5·0.9^步數`（成功）或 0（失敗） |
