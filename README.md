# csp-aps

光热塔式电站（圆柱外置集热器）的**镜场自动布置**与**实时瞄准点策略（APS）**优化工具。自用、非商业。

需求说明见 Claude Docs 文档《光热镜场自动布置与实时瞄准策略（APS）优化算法——需求说明》。

## 当前进度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1a | 太阳位置、晴空 DNI、余弦、反射率、大气衰减、HFLCAL 光斑投影到圆柱面、截断效率、通量图、全年能量 | 已完成 |
| 1b | 阴影与遮挡（shading / blocking） | 待做（目前效率记为 1.0） |
| 1c | 简易蒙特卡洛光线追迹（MCRT），用于校验解析模型 | 待做 |
| 2 | APS：晴天、均匀 DNI，在通量上限约束下最大化接收功率 | 待做 |
| 3 | APS：云影非均匀 DNI、实时、瞄准点平滑 | 待做 |
| 4 | 镜场布置优化（参数化布局 + 全年评价） | 待做 |

## 安装

需要 Python ≥ 3.10。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## 使用

所有输入都在一个 YAML 算例文件里，参考 `cases/example.yaml`（虚构的测试参数）。

```bash
cspaps layout   cases/example.yaml                          # 生成镜场布置 -> layout.csv / layout.png
cspaps snapshot cases/example.yaml --time "2025-06-21 13:40" # 单一时刻：效率分解 + 集热器通量图
cspaps annual   cases/example.yaml                          # 全年：代表日 × 逐时累计能量
```

时间按算例里的 `site.timezone` 解释为当地时间。`snapshot` 可加 `--dni 800` 覆盖 DNI。结果写到 `results/<算例名>/`：

- `*.json`：汇总（各环节功率/能量、效率、峰值通量）
- `*_heliostats.csv`：每面镜子的坐标、瞄准点和各项效率
- `*_flux.png`：圆柱面展开通量图（横轴为集热器方位，青色线圈出超过通量上限的区域）
- `*_efficiency.png`：镜场效率分布

示例算例（1532 面 8×8 m 镜子，北纬 40°）的结果：

| 指标 | 夏至 13:40 | 全年（单点瞄准） |
| --- | --- | --- |
| 余弦效率 | 0.834 | 0.772 |
| 反射率 × 清洁度 | 0.912 | 0.912 |
| 大气衰减 | 0.961 | 0.961 |
| 截断效率 | 0.962 | 0.956 |
| 总光学效率（未计阴影遮挡） | 0.703 | 0.647 |
| 峰值通量 (kW/m²) | 1199 | 1237（上限 1000，全年约 2800 h 超限） |

单点瞄准会超出通量上限，这就是需要 APS 的原因。

### 用自己的镜场坐标

在 YAML 里把 `field.generator` 换成 `field.layout_file: my_layout.csv`。CSV 表头为 `x,y` 或 `x,y,z`，路径相对 YAML 文件。

## 模型说明

坐标系：x = 东，y = 北，z = 上，单位 m，原点在塔底；方位角按罗盘计（北 = 0°，东 = 90°，顺时针）。

每面镜子的功率链：

```
P_sun → ×余弦 → ×反射率·清洁度 → ×阴影遮挡 → ×大气衰减 → ×截断 → 到达集热器表面
```

- **太阳位置**：NREL SPA（pvlib）。**DNI**：Meinel 晴空模型 + 海拔修正，或常数。
- **大气衰减**：Vittitoe & Biggs 晴天公式（随斜距）。
- **光斑**：HFLCAL 圆高斯模型，σ² = 太阳形状² + (2·斜率误差)² + (2·跟踪误差)² + 像散²，像散项按聚焦方式（斜距聚焦 / 平面镜 / 固定焦距）计算。
- **投影到圆柱面**：沿主光线把集热器表面点投影到像平面，乘以入射余弦；背光面为 0。
- **截断效率**：对圆柱侧面在像平面上的轮廓做一维 Gauss-Legendre 积分，与通量图网格精度无关。通量图积分应与之一致（`flux_map_power_mw` 字段可用来检查）。

## 开发

```bash
pytest -q
```

代码结构：

```
src/cspaps/
  config.py    算例 YAML -> dataclass（拼错的键会报错）
  sun.py       太阳位置、DNI
  receiver.py  圆柱面网格、瞄准点几何
  field.py     镜场：径向交错布置生成、边界裁剪、CSV 读写
  optics.py    余弦、衰减、光斑 σ、通量叠加、截断效率
  simulate.py  单时刻 snapshot、全年 annual
  plotting.py  布置图、通量图
  cli.py       命令行
tests/         解析解对比、能量守恒、端到端
```
