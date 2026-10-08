<div align="center">

[English](README.md) | **繁體中文**

# PRISM: Physics-Radio Integrated Simulation and Management for V2X Digital Twins

**Chia-Chuan Chiu<sup>\*</sup>, Ming-Chun Lee<sup>\*</sup>, Yung-Sheng Chao<sup>†</sup>, Li-Chun Wang<sup>\*</sup>**

<sup>\*</sup>國立陽明交通大學 電機學院<br>
<sup>†</sup>國家中山科學研究院

**IEEE VTC2026-Spring**

[📄 論文 (PDF)](PRISM_VTC2026_Spring.pdf) · [📝 引用](#引用)

</div>

---

**PRISM** 是一個為車聯網（V2X）數位孿生打造的物理與無線電整合模擬與管理平台。它用 [Sionna](https://nvlabs.github.io/sionna/) 光線追蹤模擬無線傳播，用 [CARLA](https://carla.org/) 模擬車輛動態，兩者同步進行共同模擬（co-simulation）。各模組之間以受 MCP 啟發、人類可讀的 JSON 指令溝通。

<p align="center">
  <img src="fig/architecture.png" width="95%" alt="PRISM 系統架構">
  <br><em>PRISM 系統架構。</em>
</p>

## 主要特色

- **多模態共同模擬架構**：模組化框架讓 CARLA 的車輛移動與 Sionna 的光線追蹤保持同步，可產生時空對齊的多模態資料，包括 RGB 影像、深度、語意分割、移動軌跡與無線通道。
- **情境驅動執行（Context-driven execution）**：受 MCP 啟發的控制介面，透過 UDP 傳送結構化 JSON 指令。指令容易閱讀，也方便和 LLM 整合；架構支援分散式部署、執行期重新配置，以及外掛外部模組。
- **梯度式天線場型重建**：利用 Sionna 的可微分光線追蹤，只需要少量量測資料就能學出天線場型，不需要完整的天線規格也能縮小 sim-to-real 差距。

## Demo

### CARLA ↔ Sionna 共同模擬

CARLA 負責更新車輛位置並渲染物理環境，Orchestrator 把每個物件的位置轉送給 Sionna，Sionna 再針對同一個時間點執行光線追蹤。

<p align="center">
  <img src="fig/cosim_demo.png" width="90%" alt="CARLA 與 Sionna 共同模擬">
</p>

### 時空對齊的多模態資料

<p align="center">
  <img src="fig/multimodal_data.png" width="95%" alt="多模態資料產生">
</p>

### 透過外部控制模組進行閉迴路波束選擇

路側單元（RSU）有三個 28 GHz 波束（半功率波束寬 30°，方位角間隔 15°），服務一台沿紅色箭頭移動的車輛。無線環境會把 TX/RX 位置持續傳給**外部控制模組**，由外部模組選出最佳波束後回傳，核心模擬流程完全不用修改。

<p align="center">
  <img src="fig/beam_selection_setup.png" width="48%" alt="波束選擇場景">
  <img src="fig/beam_selection_result.png" width="50%" alt="波束選擇結果">
</p>

前 15 秒維持預設的 beam 1。開啟波束選擇後，外部模組在第 17 秒切換到 beam 2、第 23 秒切換到 beam 3，讓車輛在剩下的時間都維持在最佳路徑增益。

### 天線場型重建

TX 天線場型建模為 $G(\theta,\phi)=\left|\sum_{k=0}^{N-1} a_k \cos(k\theta)\right|$，其中 $a_k$ 是可訓練係數。我們用室內場景的 1,000 筆接收功率資料，透過可微分光線追蹤反向傳播來擬合這些係數。重建的場型在室內達到 **1.6 dB RMSE**，套用到室外的陽明交大校園地圖時仍維持約 **3 dB**。

| 目標場型 | 初始場型 | Epoch 10 | Epoch 50 | Epoch 100 |
|:---:|:---:|:---:|:---:|:---:|
| <img src="fig/antenna_target.png" width="160"> | <img src="fig/antenna_initial.png" width="160"> | <img src="fig/antenna_epoch10.png" width="160"> | <img src="fig/antenna_epoch50.png" width="160"> | <img src="fig/antenna_epoch100.png" width="160"> |

### 場景

| 陽明交大校園（約 1200 m × 1200 m，取自 OpenStreetMap） | 室內地圖（用於天線場型訓練） |
|:---:|:---:|
| <img src="fig/nycu_campus.png" width="420"> | <img src="fig/indoor_map.png" width="360"> |

### 運算時間

測試硬體：Intel Ultra 7 265K、NVIDIA RTX 5070 Ti（16 GB）、32 GB RAM。

| CARLA 每步時間 vs. 車輛數 | 光線追蹤時間 vs. RX 數（depth = 5） | 光線追蹤時間 vs. depth（1 TX） |
|:---:|:---:|:---:|
| <img src="benchmarks/carla_time_vs_num_vehicle.png" width="300"> | <img src="benchmarks/rt_time_vs_num_rx.png" width="300"> | <img src="benchmarks/rt_time_vs_num_depth.png" width="300"> |

- 車輛數從 1 增加到 40 時，物理環境每個 0.1 秒的模擬步需要 **9 到 16 ms**。
- 在 28 GHz 下，1 TX / 1 RX 的光線追蹤**不到 30 ms**，8 TX / 100 RX 約 **1 秒**。增加光線追蹤深度對運算時間的影響很小。

---

## 專案結構

```
.
├── run_scheduler.py               # Orchestrator：載入情境並轉送 JSON 指令
├── run_wireless_env.py            # 無線環境：Sionna 光線追蹤 + 鏈路層模擬
├── run_visualizer.py              # 物理環境：CARLA client、感測器、資料集寫入
├── run_external_module.py         # 外部控制模組（閉迴路波束選擇範例）
├── simulation_config_with_carla.json  # 初始配置 / 情境（JSON 指令列表）
├── utils.py                       # NetworkComponent：UDP + JSON 通訊基礎類別
├── commu_modular.py               # LDPC / OFDM 鏈路層模擬（Sionna PHY）
├── dataset_writer.py              # 非同步多模態資料集寫入器
├── example_usage.py               # DatasetWriter 使用範例
├── path_generator.py              # 由關鍵點產生 RX 移動軌跡的工具
├── custom_antenna/                # 自訂與可訓練天線場型（場型重建 notebook）
├── visualize_web/                 # Flask 網頁儀表板（即時 SNR 圖 + 場景渲染）
├── my_scene/
│   ├── nycu_campus/               # 陽明交大校園的 Sionna 場景（Mitsuba XML + meshes）
│   ├── nycu_carla_v7_add_plane_yf_mirror/  # 與 CARLA 地圖對齊的 Sionna 場景
│   └── carla_format/              # CARLA 地圖素材（FBX、OpenDRIVE .xodr、貼圖）
├── benchmarks/                    # 論文中運算時間與波束選擇結果的腳本 / notebook / 圖
├── debug/                         # 開發與除錯用的小工具
└── PRISM_VTC2026_Spring.pdf       # 論文
```

### 論文與程式的名稱對照

| 論文中的模組 | 程式 | 元件名稱 / UDP port |
|---|---|---|
| Orchestrator | `run_scheduler.py` | `scheduler` / 5000 |
| Wireless Environment（Sionna） | `run_wireless_env.py` | `wireless_env` / 5001 |
| Physical Environment（CARLA） | `run_visualizer.py` | `visualizer` / 5003 |
| External Control Module | `run_external_module.py` | `external_module` / 5004 |
| 網頁儀表板 | `visualize_web/app.py` | HTTP 5080 |

## 快速開始

### 環境需求

- Linux 與 NVIDIA GPU
- Python 3
- [Sionna](https://nvlabs.github.io/sionna/)（Sionna RT + Sionna PHY，含 Mitsuba / Dr.Jit）與 TensorFlow
- [CARLA](https://carla.org/) 模擬器與其 Python API
- 其他 Python 套件：`numpy`、`matplotlib`、`opencv-python`、`pillow`、`flask`、`filelock`、`pandas`、`scikit-learn`

> **TODO：** 補上實際測試過的 Python、Sionna、TensorFlow、CARLA 版本。

如果 CARLA Python API 不是用 pip 安裝的，請把環境變數 `CARLA_ROOT` 設成 CARLA 的安裝目錄，讓 `run_visualizer.py` 找得到 `.egg`。

### 在 CARLA 載入陽明交大地圖

`my_scene/carla_format/nycu_carla_v6/` 裡是 CARLA 地圖素材，包含 FBX 模型、OpenDRIVE `.xodr` 和貼圖。執行共同模擬前，請先把它們匯入 CARLA 成為自訂地圖。對應的 Sionna 場景是 `my_scene/nycu_carla_v7_add_plane_yf_mirror/`。

### 執行

在 repo 根目錄下，每個模組各開一個 terminal 執行：

```bash
# 0. 啟動 CARLA server（並載入陽明交大地圖）

# 1. 無線環境（Sionna）
python run_wireless_env.py

# 2. 網頁儀表板（可選）與物理環境（CARLA）
python visualize_web/app.py
python run_visualizer.py

# 3. （可選）外部控制模組，例如閉迴路波束選擇
python run_external_module.py

# 4. Orchestrator：載入 simulation_config_with_carla.json 並開始模擬
python run_scheduler.py
```

用瀏覽器開啟 `http://localhost:5080` 即可查看即時 SNR 與場景渲染。

在配置中啟用 `set_dataset_writer` 後，產生的資料（影像 / 深度 / 語意分割 / LiDAR / 通道）會寫到 `./dataset/<scenario_id>/`。

## 情境驅動指令

每個模組都用 JSON 指令控制，指令中會指定目標模組、操作類型和參數。因為每道指令都帶有物件座標與模擬時間戳，物理環境和無線環境能自然保持時空一致。

```jsonc
// 移動接收端
{
  "target": "wireless_env",
  "message_type": "update_object_position",
  "ID": "rx01",
  "position": [10, 20, 1.5]
}
// 針對此時間點觸發光線追蹤
{
  "target": "wireless_env",
  "message_type": "run_ray_tracing",
  "timestamp": 1
}
```

> 上面的指令沿用論文中的寫法。程式中實際的 `message_type` 名稱是 `initialize`、`create_object`、`update_object`、`run`、`update_antenna_pattern`、`set_dataset_writer`、`update_world` 等，請參考各個 `run_*.py` 裡的 `register_handler(...)`，以及 `simulation_config_with_carla.json` 的範例情境。

一個情境就是一串這樣的 JSON 指令，Orchestrator 會在啟動時依序執行。外部程式（包括 AI agent 或 LLM）也能送出同樣的指令，在執行期間重新配置模擬。

---

## 引用

如果 PRISM 對你的研究有幫助，請引用：

```bibtex
@inproceedings{chiu2026prism,
  title     = {{PRISM}: Physics-Radio Integrated Simulation and Management for {V2X} Digital Twins},
  author    = {Chiu, Chia-Chuan and Lee, Ming-Chun and Chao, Yung-Sheng and Wang, Li-Chun},
  booktitle = {Proc. IEEE 103rd Vehicular Technology Conference (VTC2026-Spring)},
  year      = {2026},
  pages     = {TBD},
  doi       = {TBD}
}
```

> 正式的 BibTeX 會在論文集出版後更新。


PRISM 建構於 [Sionna](https://github.com/NVlabs/sionna) 與 [CARLA](https://github.com/carla-simulator/carla) 之上。地圖資料 © [OpenStreetMap](https://www.openstreetmap.org/copyright) contributors，採用 [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/) 授權。

## 授權

本 repo 的程式碼採用 [MIT License](LICENSE) 授權。

論文 PDF 為作者的 accepted manuscript，版權聲明以英文原文為準：© 2026 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works.
