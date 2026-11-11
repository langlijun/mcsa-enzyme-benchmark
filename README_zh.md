# M-CSA酶复合物Benchmark

这是已审20例评价器`0.7.0`的独立代码仓库，不运行AF3等结构预测模型。

## 公开内容与分层

2026-11-10快照提供MIT许可代码、[20例模型中立输入](benchmark/tasks)、
冻结判据、[任务复杂度分层v1](docs/task_complexity.md)和
[逐例五样本基线计数](benchmark/baseline_counts.csv)。
分层为2例Easy、11例Medium、7例Hard，依据任务中声明的位点及化学关系组织要求，
由作者逐例定义；不是已独立验证的模型预测难度，也不以本批通过率划分。
MSA、模板及化学状态参考另列。此次不改变评分版本、输入化学状态、判据或结果。

## 两阶段分开

Mechanism检查所选M-CSA步骤的反应原子距离、方向、特定位点及金属组织。
General独立检查输入立体化学、已有键几何、平面性、非金属短接触及体积重叠等。
Joint只是同一样本两阶段都通过的交集。

**金属距离下限不计入General。** 已批准的case专属mechanism配位窗口不变；
1L7N没有Mg距离下限。普通非金属碰撞不能因为机制通过而豁免，只有明确登记的
case专属规则例外。当前General不是未经修改的PoseBusters，也不计算能量。

## 开始使用

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install .
mcsa-benchmark list-cases
mcsa-benchmark criteria --case 1L7N
mcsa-benchmark evaluate prediction.cif --case 1L7N --json result.json
```

每个case都要求其声明的组装体、配体数量和化学状态。2GSA必须提供
`--atom-profile AF3`等已定义profile或完整atom map；1LDM的主任务是四聚体与四套组分。
不能把不匹配的输入直接作为同一任务计分。

批量评价：

```bash
mcsa-benchmark-panel --manifest panel.json --data-root ./data_private --output ./results/run1
```

每个case/model的五份样本必须带原生排名。Top1取rank1，不是固定sample0；
Best5要求至少一份完整样本通过；联合Best5不允许跨样本拼接。

## 测试与复现

从仓库根目录直接评价仓库内的400份预测，不需要其他数据目录：

```bash
mcsa-benchmark-panel --manifest benchmark/prediction_manifest400.json --data-root . --output results/benchmark20
```

```bash
python tools/run_tests.py
python tools/run_tests.py --fixtures /path/to/reviewed-data-archive
```

无数据时明确列出跳过的结构测试；提供完整内部归档时要求全部运行且无跳过。
运行时不读取历史成败表。批量回归工具可将新计算的机制、General和原子映射
与冻结结果逐项比较，而不只是比较通过数量。

## 发布范围

本仓库包含代码、标准任务输入、400份预测CIF、原生排名、分层与汇总结果。
坐标保存在benchmark/predictions，路径及排名保存在benchmark/prediction_manifest400.json，
所有数据路径均相对于仓库根目录。不包含Excel、文献PDF、个人路径或服务器凭据。
可选的历史扰动测试夹具与这400份主样本不同，不是运行上述评价命令的依赖。
原创代码使用MIT许可；第三方资料及AF3派生结果不因此改用MIT，
详见[许可与数据范围](docs/data_policy.md)和[第三方声明](THIRD_PARTY_NOTICES.md)。

20例判据和文献依据见[case目录](docs/cases.md)，使用说明见[英文README](README.md)，
方法范围见[Methods](docs/methods.md)，批量格式见[复现说明](docs/reproduction.md)。
判据属于文献支持的操作性筛查，不等于实验活性阈值；本工作具有回顾性，
不能表述为预注册或已证实训练集独立。

