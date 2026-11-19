# csp-aps

光热塔式电站（圆柱外置集热器）**镜场自动布置**与**实时瞄准点策略（APS）**优化软件。自用、非商业。

输入镜场最大范围、集热器尺寸和中心点坐标、定日镜尺寸，软件计算各项光学损失，给出：

- **内层 APS（实时控制引擎）**：在当前太阳位置和每面镜子的实际 DNI（含云影）下，算出每面镜子在圆柱集热器上的瞄准点，使接收功率最大，同时不超过通量上限、受热尽量均匀。秒级求解，支持 DNI 预报、瞄准点平滑和散焦。
- **外层布置优化（离线规划引擎）**：在场地边界内用参数化布局（径向交错 / 仿生螺旋）生成候选位置，按全年能量选出指定数量的镜子，用差分进化做全局寻优，最后对前几名方案用完整 APS 精算全年能量。
- **全年能量**：用 20–40 个代表太阳位置做加权积分（误差 < 0.5%），也可以逐时或用代表日计算。
- **MCRT 离线校验**：独立的蒙特卡洛光线追迹，用来检验解析模型的精度。

模型细节见 [docs/MODEL.md](docs/MODEL.md)。

## 安装

需要 Python ≥ 3.10。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gui]"
```

开发（含测试）用 `pip install -e ".[dev]"`。

## 使用

### 图形界面

```bash
cspaps gui
```

浏览器会自动打开本地页面，各标签页的内容：

| 标签页 | 内容 |
| --- | --- |
| 参数设置 | 全部输入参数；可上传镜场坐标 CSV、气象文件；可载入或保存算例 YAML |
| 镜场布置 | 生成镜场布置图 |
| 瞄准优化 APS | 选择日期、时刻和方法，算出瞄准点、效率分解、通量图；可上传实测 DNI 分布 |
| 云影实时仿真 | 移动云影下三种策略的对比 |
| 全年能量 | 全年能量与年均效率 |
| 布置优化 | 外层优化；可一键把优化结果设为当前镜场 |
| MCRT 校验 | 解析模型与光线追迹的对比 |
| 代表太阳位置 | 全年代表太阳位置及其误差 |

所有结果都可以在页面上下载，同时保存在 `results/gui/`。

### 命令行

```bash
cspaps layout          cases/example.yaml                              # 生成镜场
cspaps snapshot        cases/example.yaml --time "2025-06-21 13:40"    # 单一时刻：效率分解 + 通量图（中心瞄准）
cspaps aps             cases/example.yaml --time "2025-06-21 13:40"    # 单一时刻：APS 瞄准点
cspaps realtime        cases/example.yaml                              # 云影下的实时 APS 仿真
cspaps annual          cases/example.yaml [--method hourly] [--aiming greedy]
cspaps optimize-layout cases/example.yaml                              # 镜场布置优化
cspaps validate        cases/example.yaml --time "2025-06-21 13:40" --rays 3000   # MCRT 校验
cspaps sunbins         cases/example.yaml --check                      # 代表太阳位置及其误差
```

时间按算例中的 `site.timezone` 解释为当地时间。结果写在 `results/<算例名>/`，包括 JSON 汇总、CSV 明细和 PNG 图。

### 输入

所有参数都在一个 YAML 文件里，`cases/example.yaml` 有全部参数和中文注释（虚构的测试参数，不对应任何真实电站）。拼错的参数名会直接报错，不会被悄悄忽略。

| 段 | 内容 |
| --- | --- |
| `site` | 经纬度、海拔、时区 |
| `receiver` | 圆柱集热器中心点、直径、高度、允许最大通量、通量网格、塔身直径 |
| `heliostat` | 镜子尺寸、旋转中心高度、反射率、清洁度、斜率误差、跟踪误差、聚焦方式 |
| `sun` / `atmosphere` / `weather` | 太阳形状、晴空 DNI 模型；大气衰减模型；气象文件（EPW 或 CSV） |
| `shading` | 阴影遮挡的采样精度、塔影 |
| `field` | 场地边界多边形、禁排半径、布置生成器参数，或直接给镜场坐标 CSV |
| `aps` | 求解方法、安全裕度、候选瞄准点、移动惩罚、散焦 |
| `realtime` | 实时仿真时段、刷新周期、预报时长、云影 |
| `annual` | 全年计算方式、代表太阳位置数 |
| `layout_opt` | 镜子数量、被优化的参数及范围、进化代数、并行进程数 |
| `mcrt` | 光线数 |

外部数据文件：

| 文件 | 格式 | 用途 |
| --- | --- | --- |
| 镜场坐标 | CSV，表头 `x,y` 或 `x,y,z` | `field.layout_file` |
| 气象数据 | EPW，或 CSV 表头 `time,dni`（当地时间） | `weather.file`，用于全年计算 |
| 实测 / 预报 DNI 分布 | CSV，表头 `x,y,dni` | `aps --dni-map` / `--forecast-dni-map`，插值到每面镜子 |
| 瞄准点 | `aps` 命令输出的 `*_aims.csv` | 下一周期的 `--prev-aims`（平滑移动） |

### 接入电站控制循环

每个刷新周期调用一次：

```bash
cspaps aps plant.yaml --time "2025-06-21 13:40" \
    --dni-map dni_now.csv --forecast-dni-map dni_max_next_2min.csv --prev-aims last_aims.csv
```

输出 `*_aims.csv`，每面镜子一行：`id, x, y, z, active, aim_azimuth_deg, aim_z, aim_x, aim_y`，以及各项效率、DNI 和功率。其中 `aim_x, aim_y, aim_z` 是瞄准点在集热器表面上的三维坐标，`active = 0` 表示散焦。

也可以在 Python 里直接调用：

```python
from cspaps import load_case
from cspaps.field import build_field
from cspaps.aps import optimize_aims

case = load_case("plant.yaml")
field = build_field(case)
res = optimize_aims(case, field, sun_elevation, sun_azimuth,
                    dni=dni_per_heliostat,                 # 当前 DNI（每面镜子）
                    dni_constraint=dni_forecast_max,       # 预报时段内的最大 DNI
                    prev=last.aim, prev_active=last.active)
aim_azimuth, aim_z = res.aim                               # 集热器方位角 (°) 与高度 (m)
```

## 示例结果

以下结果来自 `cases/example.yaml`，是虚构的测试参数：北纬 40°，海拔 1200 m，晴空 DNI；集热器直径 7 m、高 8 m，中心高 100 m，通量上限 1000 kW/m²；1532 面 8×8 m 定日镜。在 4 核机器上运行。

### 内层 APS：单一时刻（夏至 13:40）

| 方法 | 集热器功率 (MW) | 峰值通量 (kW/m²) | 周向不均匀度 (CV) | 求解时间 |
| --- | --- | --- | --- | --- |
| 全部瞄准中心 | 69.13 | 1191（超限） | 0.197 | — |
| k 因子法 | 68.98 | 949 | 0.197 | 0.3 s |
| 贪心法（默认） | 69.03 | 882 | 0.144 | 1.1 s |

求解前需要预计算候选点的光斑，约 1.8 s。贪心法只损失 0.15% 的功率，就把峰值从 1191 降到 882 kW/m²（允许值为 950，即上限扣除 5% 安全裕度），周向也更均匀。

### 云影下的实时运行（20 分钟，每 60 s 刷新，4 朵随机云）

| 策略 | 20 分钟能量 (MWh) | 最大峰值通量 (kW/m²) | 超限时长 | 平均每次调整的镜子数 | 平均求解时间 |
| --- | --- | --- | --- | --- | --- |
| 不做 APS | 23.19 | 1198 | 21 分钟 | 0 | — |
| APS，只用当前 DNI | 23.15 | 1061 | 1 分钟 | 242 | 3.0 s |
| APS + DNI 预报 | 23.15 | 879 | 0 | 152 | 3.0 s |

只用当前 DNI 的 APS 在云移走、DNI 回升的那一刻会超通量；加入预报后不再超限，能量不变，调整的镜子还更少。

### 全年能量

30 个代表太阳位置，计算约 3.5 s：

| 损失环节 | 余弦 | 反射率×清洁度 | 阴影 | 遮挡 | 大气衰减 | 截断 | 总光学效率 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 年均效率 | 0.773 | 0.912 | 0.970 | 0.994 | 0.961 | 0.956 | 0.624 |

| 计算方式 | 年能量 (MWh) | 超通量小时 |
| --- | --- | --- |
| 逐时全年（基准，约 3.5 分钟） | 206,420 | — |
| 30 个代表太阳位置，中心瞄准 | 207,252（+0.40%） | 2537 |
| 30 个代表太阳位置，每个时刻做 APS（97 s） | 206,923 | 0 |

### 外层布置优化（固定 1200 面镜子）

- 差分进化 6 代，加 4 轮局部微调，共 134 次布局评价，用时 136 s（4 进程）。
- 优化后全年能量 173,850 MWh（中心瞄准），比示例原布置参数（同样选 1200 面）的 171,185 MWh 高 1.6%。
- 用 APS 精算的全年能量为 173,494 MWh。
- 输出的 `case_optimized.yaml` 可以直接作为新算例使用。

### MCRT 校验（每面镜子 3000 条光线，约 5–10 s）

| 时刻 | 接收功率差 | 截断效率差 | 遮挡效率差 | 峰值通量（解析 / MCRT） | 通量图相关系数 |
| --- | --- | --- | --- | --- | --- |
| 夏至 13:40，中心瞄准 | −0.95% | −0.008 | −0.001 | 1191 / 1208 | 0.999 |
| 春分 10:00，APS 瞄准 | −0.65% | −0.005 | −0.001 | 708 / 758 | 0.998 |

余弦、反射和衰减与 MCRT 完全一致。解析模型的截断效率略偏保守（低 0.5–0.8%），这是 HFLCAL 圆高斯近似带来的。

## 开发

```bash
pip install -e ".[dev]"
pytest -q
```

测试的重点：
- 解析解对比（截断效率、平行镜阴影、塔影）；
- 能量守恒；
- MCRT 一致性；
- APS 约束满足，以及与 MILP 精确解、LP 上界的对比；
- 命令行和图形界面的端到端运行。

代码结构：

```
src/cspaps/
  config.py      算例 YAML -> dataclass（含校验）
  sun.py         太阳位置、DNI、气象文件、全年采样与代表太阳位置
  receiver.py    圆柱面网格与瞄准点几何
  field.py       镜场生成（径向交错 / 仿生螺旋）、边界、CSV
  optics.py      余弦、大气衰减、HFLCAL 光斑、截断效率、通量叠加
  shading.py     阴影、遮挡、塔影
  simulate.py    单时刻功率链、全年能量
  aps.py         内层：瞄准点优化（center / kfactor / greedy / milp）
  clouds.py      云影与实测 DNI 分布
  realtime.py    实时控制循环仿真
  layout_opt.py  外层：镜场布置优化
  mcrt.py        蒙特卡洛光线追迹校验
  workflows.py   各任务的计算与输出（命令行和图形界面共用）
  plotting.py    图
  cli.py         命令行
  gui_app.py     图形界面
docs/MODEL.md    模型与算法说明
```
