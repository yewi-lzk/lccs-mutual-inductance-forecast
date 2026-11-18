# LCC-S 无线传能系统动态互感的短时预测

《LCC-S无线传能系统动态互感的短时预测》（V19）的图表、逐轨迹结果与复核代码。研究问题是，线圈运动时，利用截至当前的五频原边电流，能多准确地预测 10 ms 和 30 ms 后的互感？

五频原边电流先经非线性最小二乘（NLS）辨识当前参数。各预测方法使用同一观测条件，GRU 采用 32 个采样点的历史，观测间隔为 5 ms。评分使用目标时刻的 69 kHz 电磁地图互感。主测试含 120 条 Simulink 动态观测轨迹，分布内（ID）与运动分布外（OOD）各 60 条。OOD 指测试时采用不同的运动条件，两组使用相同的电磁地图。

在 1% 电流相量噪声、30 ms 预测时域的运动分布外测试中，直接保持当前互感估计的平均绝对百分比误差（MAPE）为 1.9288%，GRU 为 1.5659%，相对降低 **18.82%**。对应的平均绝对误差（MAE）从 **0.368 μH** 降至 **0.298 μH**。文中的互感预测准确率定义为 `100% − MAPE`。不同方法的排序随预测时长和运动条件而变，完整比较见表 3。

## 从哪里看起

- 查论文中的数值，可直接打开[附件 S1](S1.xlsx)。按轨迹计算的数据也以 CSV 提供，可直接比较同一轨迹上的两种方法。
- 查看五频 NLS、GRU 与对照模型的实现，可从[算法说明与冻结模型示例](model/README.md)进入。核对示例轨迹时，将 Zenodo 第 01、03、04 类数据包解压到同一父目录，分别读取电磁参数、电路配置和已存的模型权重与预测结果。
- 用 Python 3.9 或更新版本运行 `python scripts/verify.py`，仅需标准库。脚本从逐轨迹结果复核两种时域的误差、GRU 的相对降幅及胜出轨迹数，核对图 7 的时间对齐，并从 750 次原始计时记录复算表 5 的分位数。[统计复核代码](scripts/verify.py)
- 从仓库根目录运行下列命令，可从 CSV 重绘图 4、6、7、8。输出为英文标注的 PNG，论文中的中文原图保存在 `figures/`。[绘图代码](scripts/plot.py) · [依赖](requirements.txt)

```sh
python -m pip install -r requirements.txt
python scripts/plot.py --output outputs/figures
```

## 论文里的每幅图

| 论文原图 | 图回答的问题 | 对应数据或材料 |
|---|---|---|
| [图 1　LCC-S 电路](figures/fig01-circuit.png) | 从电路的哪一侧采集电流？互感位于哪个环节？ | [标称电路参数](data/checks/nominal_circuit.csv) |
| [图 2　线圈几何与网格](figures/fig02-coils-mesh.png) | 位置到复阻抗的电磁地图源自什么线圈模型？ | 论文所用 COMSOL 几何与网格截图，原生模型在六类归档中的第 01 类 |
| [图 3　动态观测与评分参考](figures/fig03-observation.png) | 怎样由同一条运动轨迹生成原边观测，并在目标时刻取参考互感？ | 流程示意图，运动参数和动态观测分别在第 02、03 类 |
| [图 4　原边电流](figures/fig04-current.png) | 代表性运动中的电流包络和末段 100 μs 载波是什么形态？ | ID [包络](data/figures/fig04_id_envelope.csv) / [载波](data/figures/fig04_id_carrier.csv)，OOD [包络](data/figures/fig04_ood_envelope.csv) / [载波](data/figures/fig04_ood_carrier.csv) |
| [图 5　GRU 结构](figures/fig05-gru.png) | 模型如何从一段辨识历史直接给出多个未来时刻的预测？ | 输入、门控循环层与输出的结构示意图，训练记录与权重在第 04 类 |
| [图 6　当前互感辨识](figures/fig06-nls.png) | 预测开始前，NLS 对当前互感的辨识效果如何？ | [当前估计与参考曲线](data/figures/fig06_current_nls.csv) · [单个五频观测包](data/checks/nls_packet.csv) |
| [图 7　GRU 预测跟踪](figures/fig07-forecast.png) | 在同一条运动轨迹上，10 ms 和 30 ms 预测各自如何跟踪目标互感？ | [两条轨迹、两种时域的 1,460 行对齐数据](data/figures/fig07_gru_forecast.csv)，`origin_s` 为用于本次预测的末次观测时刻，`target_s` 为评分时刻 |
| [图 8　时域与噪声](figures/fig08-sensitivity.png) | 延长预测时域或改变电流观测噪声后，误差怎样变化？ | [分预测时域](data/tables/forecast_by_horizon.csv) · [10 ms 分噪声水平](data/tables/forecast_by_noise_at_10ms.csv) |

## 论文里的每张表

| 论文表格 | 汇总与复核依据 |
|---|---|
| 表 1　数据划分与仿真核对 | [轨迹分组、种子及复用关系](data/tables/trajectory_splits.csv) · [载波电路核对](data/checks/physical_crosscheck.csv)。48 条训练和 12 条验证轨迹采用准静态观测。主测试的 120 条动态轨迹单独计数，24 条桥接比较与 2 条复用验证轨迹保留配对标记。 |
| 表 2　五频 NLS 同刻辨识 | [分组汇总](data/tables/nls_current.csv) · [逐轨迹辨识误差](data/checks/trajectory_nls.csv)。`common_current` 对应共同预测起点，`all_current` 对应每条轨迹的全部 401 个观测包。 |
| 表 3　六种方法的 10 ms 与 30 ms 预测 | [按时域和方法汇总](data/tables/forecast_by_horizon.csv) · [1,440 条逐轨迹 MAPE](data/checks/trajectory_mape.csv)。在汇总表筛选 `horizon_ms=10` 或 `30` 即可对应正文。 |
| 表 4　GRU 相对各基线的收益 | [配对差值、置信区间和胜出轨迹数](data/tables/gru_paired_gains.csv) · [分运动族汇总](data/tables/forecast_by_motion_family.csv)。相对降幅为 `100 × (基线 MAPE − GRU MAPE) / 基线 MAPE`。区间沿用实验归档的分层轨迹重采样结果。 |
| 表 5　桌面原型耗时 | [四步 GRU 分项计时](data/tables/gru_timing.csv) · [750 次调用记录](data/checks/gru_timing_samples.csv)。四步模型提供 10 ms 输出；该表的计时对象仅为四步实现。 |

第 2 章选取 5 ms 观测间隔的依据见[观测分辨率核验](data/checks/observation_resolution.csv)，同轨迹动态与准静态观测的比较见[桥接结果](data/checks/matched_bridge.csv)。摘要的 μH 误差可查[30 ms 逐轨迹 MAE](data/checks/mae_30ms.csv)，第 4.3 节可查[保持参考与物理误差](data/checks/reference_hold_and_mae.csv)，第 5 章的空间插值检查见[六个非网格位姿的有限元核对](data/checks/heldout_fea.csv)。

## 数据口径

每个测试域包含 6 个运动族，每族 10 条轨迹。预测时只能使用当时及以前的原边观测和辨识历史；轨迹位置与电磁地图提供评分参考。`CurrentHold` 保持当前 NLS 估计，`DenoisedHold` 保持去噪后的估计。`Linear`、`Ridge_Residual`、`TCN_Residual` 和 `GRU` 是四种未来预测方法，六种方法使用相同的域和时域进行配对比较。统计单位是运动轨迹。

10 ms 的结果取自四步模型，每条轨迹有 366 个共同预测起点。30 ms 的结果取自六步模型，每条有 364 个。模型输出步数与所选评分时刻可在时域汇总表中分别查阅。论文评价仿真观测条件下的互感预测精度，桌面计时作为实现信息单列。
