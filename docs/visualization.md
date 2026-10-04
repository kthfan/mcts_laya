# 視覺化介面

讓使用者看到 Laya + MCTS 每一步在想什麼：Laya 給了哪些先驗、搜尋走了哪些路、最後為什麼選這一步，以及整個訓練過程怎麼進步。

## 各環境的評估

| 環境 | 現成的視覺化引擎 | 適合度 | 採用的做法 | 狀態 |
|---|---|---|---|---|
| Countdown | 沒有 | 高：數字、算式、3–4 步的小搜尋樹一目了然 | HTML 前端：數字方塊 + 目標 | ✅ |
| 代數（一元一次方程） | 沒有（SymPy 只有算式排版，沒有互動介面） | 高：每一步改寫和候選步驟都能直接顯示 | HTML 前端：方程式 + 已用步數 | ✅ |
| TextWorld | **有**：`textworld.render` 把世界狀態序列化成房間座標、物件、出口（`tw-play --viewer` 用的就是它） | 高：地圖加上 agent 走過的房間，最能說明探索過程 | 用 TextWorld 引擎的 `load_state_from_game_state` 取得地圖資料，畫在同一套前端裡；目前位置加粗框，已造訪的房間上色，滑鼠移上去顯示房內物件 | ✅ |
| ALFWorld | 文字模式沒有；視覺版是 AI2-THOR 3D 模擬器（需要 GPU 與顯示環境，這個容器跑不動） | 中：文字版可沿用 TextWorld 的做法（ALFWorld 建在 TextWorld 引擎上，但它的遊戲由 PDDL 定義，`textworld.render` 能否直接使用要實測）；3D 畫面有助理解，但只能在 GPU 環境產生 | 先實作 `render_data` 回傳文字與（可行時）地圖；3D 畫面列為選配，之後以 AI2-THOR 截圖嵌入回放 | ⬜ Phase 1b |

設計原則：前端只認得 `render_data()` 回傳的 `kind`，新環境只要實作這個方法，就能沿用回放、搜尋樹、同題比較、學習曲線四個頁面。沒有專屬面板的環境會退回顯示狀態文字。

## 兩種使用方式

```bash
# 靜態報告：一個自帶資料的 HTML 檔，任何瀏覽器都能直接開，不需要伺服器
.venv/bin/mcts-laya viz --run runs/phase0/countdown-laya-cpu-smoke --problems 4
#   -> runs/phase0/countdown-laya-cpu-smoke/viz.html（可用 --out 指定）

# 即時模式：本機網頁，自己出題、一步步看 Laya + MCTS 搜尋，或自己點選動作
.venv/bin/mcts-laya serve --run runs/phase1a/l1-laya-cpu --port 8765
#   -> 打開 http://127.0.0.1:8765
```

兩者都可以改用 `--config <yaml>` 從設定檔建立（這時沒有學習曲線），也可以用 `--checkpoint` 指定要載入的模型。預設會使用實驗目錄裡的 `checkpoints/final`。

已產生的報告放在 [`docs/viz/`](viz/)。

## 四個頁面

| 頁面 | 內容 |
|---|---|
| 對局回放 | 選題目和方法，逐步播放（也可以用鍵盤 ← →）。左邊是環境畫面，右邊是每個候選動作的 Laya 先驗 P(s,a)、MCTS 搜尋策略 π、拜訪次數、Q 值，以及 Laya 價值 V(s) 和搜尋價值。即時模式下可以「搜尋這一步」「執行搜尋建議」「自動下完」，也可以直接點候選動作自己下 |
| 搜尋樹 | 某一步的 MCTS 樹：圓的大小與連線粗細代表拜訪次數，顏色代表 Q（藍高、灰中性、紅低），◆ 代表終局（綠成功、紅失敗）。點選節點會顯示動作路徑、先驗、Q、價值與該狀態的文字。只保留被拜訪過的節點，每層最多 8 個子節點，總數有上限 |
| 同題比較 | 同一題下各方法的結果、reward、步數和完整走法：greedy（只用 Laya、不搜尋）、Laya + PUCT / Gumbel、均勻先驗與隨機 rollout 的 MCTS 基準，以及 teacher（精確解題器，參考上限） |
| 學習曲線 | 從 `metrics.jsonl` 畫出每一輪的評估 reward 與成功率（基準以水平線表示）、訓練損失、保留資料上的 policy top-1；也可以切換成表格 |

深色 / 淺色會跟著系統設定，也可以手動切換。配色採用 dataviz 的參考調色盤：方法的顏色固定跟著方法本身，Q 值使用藍—灰—紅的發散色階。

## 資料格式（給擴充用）

`mcts_laya.viz.trace` 產生的 JSON：

```text
step = {render, text, actions: [{text, prior, policy, visits, q}], selected, chosen,
        nn_value, root_value, simulations, tree?}
tree = {a, p, n, q, v?, t?, tv?, ok?, s?, hidden, children: [tree...], size}
episode = {steps: [step...], final: {render, text}, outcome: {success, reward, moves}}
```

即時模式的 API（`mcts_laya.viz.server`）：`GET /api/state`，以及 `POST /api/new {split, index}`、`/api/search {session, method, simulations?}`、`/api/act {session, index}`、`/api/compare {session}`。伺服器只綁定 127.0.0.1，所有呼叫在同一把鎖下依序執行。
