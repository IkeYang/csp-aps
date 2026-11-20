# csp-aps

光热塔式电站（圆柱外置集热器）**镜场自动布置**与**实时瞄准点策略（APS）**优化软件。自用、非商业。

已按导师评审稿（2026-11-20，基线 0d9292d）逐条落实，对应关系见 [docs/REVIEW_RESPONSE.md](docs/REVIEW_RESPONSE.md)；模型细节见 [docs/MODEL.md](docs/MODEL.md)。

## 能做什么

| 模块 | 内容 |
| --- | --- |
| 在线 APS 控制引擎 | 每个控制周期输入时刻、逐镜实测 DNI、带有效时刻的预报快照、镜子可用状态、上一轮执行回执，输出每面镜子的瞄准坐标、散焦状态、方案有效时段和复核结果。<br>• 按管屏的通量上限<br>• 安全裕度、预报裕度<br>• 数据过期 / 缺失时回退<br>• 瞄准点平滑、可达性筛选<br>• 光斑响应缓存与热启动（热路径约 0.4 s / 1500 面镜） |
| 可行性复核 | 每个方案都在自适应细网格上用完整光斑复核：超限热点自动加入约束重解，仍不行就明确给出"不可执行"状态及热点位置、所在管屏、超限量、主要贡献镜 |
| 均匀受热 | 预热器考核区的二维 / 周向 / 高度 CV；在 CV ≤ ε 约束下最大化吸收功率；ε 扫描曲线；上下边缘功率上界（证明某些目标不可达）；小规模问题可用 SCIP 求精确最优 |
| 云影回放 | 随机云、云边扫过、云突然移开、DNI 延迟 / 丢失。对比不做 APS、只用当前 DNI、带预报三种策略：能量、超限时长（按区间统计）、端到端延迟 P50/P95、回退次数 |
| 全年能量 | 20 / 30 / 40 个 DNI 加权代表太阳位置（误差 < 0.5%）、高度角 × 方位角分箱、逐时；受限 APS 时按 DNI 分档；可读 EPW/CSV 气象文件；多进程并行 |
| 外层布置优化 | 径向交错 / 仿生螺旋参数化布局 → 差分进化 + 局部微调 → 前几名方案用受限 APS 复核并重新排名 → 换位微调 → 镜数、边界、禁排区、最小镜距最终校验（布不满时报告缺额） |
| 光学校验 | 内部 MCRT：独立误差项、多随机种子的统计不确定度、光线数收敛表。SolTrace 对标：导出同场算例和 PySolTrace 驱动脚本。遮挡筛选：与暴力求交逐镜比对 |

## 安装

需要 Python ≥ 3.10。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[gui,exact]"
```

`gui` 安装 Streamlit 和 plotly；`exact` 安装 SCIP（用于小规模精确基准，可选）。开发和测试用 `pip install -e ".[dev]"`。

## 使用

### 图形界面

```bash
cspaps gui [cases/example.yaml]
```

| 工作区 | 内容 |
| --- | --- |
| 运行调度 | 首屏状态卡：时间戳、数据来源与新鲜度、策略、截获 / 吸收功率、最高通量及所在管屏、距上限余量、散焦镜数、指令状态（待复核 / 不可执行 / 已回退 / 已执行），窄窗下自动换行。<br>• 镜场 DNI 图、接收器展开图、时间轴在同一页，拖动时间轴同步更新<br>• 点击镜子看 ID、DNI、原瞄准点、计划瞄准点、散焦原因<br>• 点击接收器看该点的限值和主要贡献镜<br>• 可回放，也可导入实测 / 预报 DNI 和上一轮瞄准点 |
| 布局规划 | 依次为：① 场地与设备（场地图标注边界、禁排区、塔位、镜位）② 气象与采样 ③ 计算（可看进度、可取消）④ 方案对比：同一表格比较各方案的镜数、全年能量、峰值、采样误差、复核方式；应用前预览，应用后自动核对镜数、编号和坐标；导出布局与参数；对比两个方案的功率–均匀度曲线 |
| 光学校验 | 解析 / 追迹 / 差值三图（同一网格、同一色标），附引擎、光线数、种子和统计不确定度；SolTrace 对标导出；遮挡筛选复核 |
| 全部参数 / 运行记录 | 全部参数表单与 YAML 编辑；每次运行的独立记录，可回看结果，也可载入当时的输入 |

### 命令行

```bash
cspaps layout          CASE                          # 生成镜场
cspaps snapshot        CASE --time "2025-06-21 13:40"  # 单时刻，中心瞄准
cspaps aps             CASE --time ... [--dni-map now.csv --forecast-dni-map max.csv --prev-aims last.csv]
cspaps realtime        CASE                          # 云影回放（场景、延迟、丢失在 YAML 中设置）
cspaps annual          CASE [--method sun_bins|sun_grid|hourly|representative_days] [--aiming center|greedy]
cspaps optimize-layout CASE                          # 布置优化 + 受限复核
cspaps uniformity      CASE --time ... [--eps 0.7 0.6 0.5] [--exact]
cspaps validate        CASE --time ... [--rays 3000 --seeds 3 --convergence 500 1000 2000]
cspaps soltrace-export CASE --time ...               # SolTrace 对标算例与驱动脚本
cspaps shading-check   CASE                          # 遮挡筛选与暴力求交对比
cspaps sunbins         CASE --check                  # 代表点误差：20/30/40 个及分箱，相对逐时
cspaps engine          CASE                          # 在线引擎：标准输入 / 输出的 JSON 行服务
```

- 每次运行写入 `results/<算例>/<任务>_<时间>/`，其中保存当时的输入（`inputs_case.yaml`）、`run.json` 和全部结果。
- 结果"不可执行"或布局校验不通过时，命令行退出码为 2。

### 接入电站控制循环

**Python 接口**：

```python
from cspaps import load_case
from cspaps.field import build_field
from cspaps.engine import APSEngine, ControlRequest, ForecastSnapshot, Receipt

case = load_case("plant.yaml")
engine = APSEngine(case, build_field(case))
resp = engine.step(ControlRequest(
    time=now, dni=dni_per_heliostat, dni_time=t_meas,
    forecasts=[ForecastSnapshot(t, dni_t) for t, dni_t in forecast],
    available=in_service, receipt=Receipt(last.plan_id, last.aim_azimuth, last.aim_z, last.active)))
resp.state            # "ready" | "fallback" | "infeasible"
resp.to_frame()       # id, active, aim_azimuth_deg, aim_z, aim_x, aim_y, defocus_reason
```

**JSON 行服务**：`cspaps engine plant.yaml`。每行一个请求，DNI 可以用列表（按镜场顺序）或 `{"镜子ID": dni}`。场站的安全联锁独立于 APS，保持不变。

### 输入

全部参数都在一个 YAML 文件里，`cases/example.yaml` 列出了每个参数并附中文注释（虚构的测试参数）。拼错的参数名会直接报错。

| 段 | 内容 |
| --- | --- |
| `site` / `sun` / `atmosphere` / `weather` | 站址、太阳形状与晴空 DNI、大气衰减、气象文件 |
| `receiver` | 圆柱集热器；管屏数、各管屏允许通量、吸收率、限值口径（入射 / 吸收）、实测 / 模型偏差系数、高度分布系数 |
| `heliostat` | 尺寸、反射率、清洁度、坡度误差、跟踪误差、镜面散射、聚焦方式 |
| `field` | 场地边界、禁排半径、布置生成器，或镜场坐标 CSV（`id,x,y,z`） |
| `aps` / `uniformity` / `engine` | 瞄准求解与复核；预热器考核区、ε、最低负荷；在线引擎周期、延迟、数据年龄、预报裕度、回退策略、转速 |
| `realtime` | 回放场景、时长、步长、预报、DNI 延迟 / 丢失 |
| `annual` / `layout_opt` | 全年采样（代表点数、分箱、DNI 分档、并行）；布置优化（镜数、参数范围、搜索与复核采样、换位、校验） |
| `mcrt` | 光线数、种子数、跟踪误差模式 |

## 示例结果

RESULTS_PLACEHOLDER

## 开发

```bash
pip install -e ".[dev]"
pytest -q        # 约 5 分钟
```

测试覆盖：

- 导师给的回归算例（`tests/test_p0_regressions.py`）；
- 解析解对比、能量守恒；
- 暴力求交与 MCRT 一致性；
- MILP 与 SCIP 精确基准；
- 在线引擎（缓存、过期数据、预报覆盖、可用状态、可达性、JSON 服务）与回放场景；
- 命令行与图形界面的端到端测试（`CSPAPS_GUI_SYNC=1` 时后台任务同步执行）。

代码结构见 `CLAUDE.md`。
