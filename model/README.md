# 五频辨识与冻结互感预测

从 59、64、69、74、79 kHz 的原边复电流出发，五频 NLS 在已知 10 Ω 负载下求当前三个对数参数，顺序为互感 M、原边自感 L₁、副边自感 L₂。五频输入单位为 A RMS。得到的三个估计分别以中心位姿的 69 kHz 标称值为参考，恢复物理量时使用 `reference * exp(theta)`。预测器取最近 32 个 NLS 估计，观测间隔 5 ms。各方法共用当前辨识结果。

| 论文环节 | 实现与方法 |
|---|---|
| 图 1、图 3、图 6、表 2 | [nls.py](nls.py) 中的 `CurrentEstimator.estimate` 用三个初值求解有界非线性最小二乘，将预测与观测的五个复电流按频率归一化，拼成十个实数残差。 |
| 图 5、图 7、表 3–4 的 GRU | [networks.py](networks.py) 中的 `Forecast` 与 `Forecast30` 使用 24 维 GRU 隐状态。冻结推断将输出按训练标准差还原，再加到当前的去噪水平。 |
| 表 3–4 的保持与 Linear | [nls.py](nls.py) 中的 `linear_prediction` 用最近 17 个估计做指数加权直线拟合。CurrentHold 保持原始 NLS 估计，DenoisedHold 保持拟合直线的当前水平，Linear 按拟合斜率外推。 |
| 表 3–4 的 Ridge 和残差 TCN | [forecast.py](forecast.py) 中的 Ridge 用训练好的线性系数修正 Linear。残差 TCN 用 [networks.py](networks.py) 中的三个因果空洞卷积块学习对 Linear 的修正，膨胀率为 1、2、4。 |

`FourStep.prepare` 与 `FourStep.predict` 对每个预测起点返回 4 个未来步长、3 个对数参数。论文 10 ms 对应数组的第 2 步。六步模型单独训练，`SixStep.predict` 返回起点索引与第 6 步、即 30 ms 的参数。返回字典按 `(方法, 种子)` 索引，确定性方法的种子为 0。两种实现都对最终参数施加同一组物理范围约束。

## 用一个已存案例核对实现

在仓库根目录运行以下命令。将 Zenodo 第 01、03、04 类解压到同一父目录，保留各类别目录名。示例中的 `../data-v19` 指这个父目录。

```sh
python -m pip install -r model/requirements.txt
python -m model.check_example "../data-v19"
```

示例只读原记录。它严格加载四步和六步的 12 组网络权重、两组 Ridge 系数及共同的归一化参数，再读取 `new_test_OOD_chirp_00` 的 1% 噪声第 0 次重复。首先核对 1.000 s 的五频包与归档 NLS 估计。随后用归档 NLS 历史对照四步全部三参数和六步末端互感，覆盖六种方法及神经网络的三个训练种子。该轨迹分别有 366 和 364 个共同预测起点。发布前在 Python 3.12 与 CPU PyTorch 2.15 的复核中，三项最大绝对对数差均为 0，脚本容差为 0.000002。全样本图表的记录与统计复核见仓库首页。

## 归档中的输入与原程序

| 类别 | 归档相对位置及用途 |
|---|---|
| 01 COMSOL 电磁建模 | `27位姿地图与原始端口记录/raw_impedance.csv` 提供标称位姿的五频复阻抗。 |
| 02 Python 运动与参数 | `运动轨迹与阻抗插值/experiment_physics.py` 包含原始电路、NLS 与线性外推；`原边电流与补偿参数计算/screen_comsol_lccs.py` 包含阻抗读取和补偿参数计算。 |
| 03 MATLAB/Simulink 动态电路 | `标称补偿电路参数/configuration.json` 提供电源与频率设置。该类另存动态观测模型与输出。 |
| 04 Python/PyTorch 辨识与预测 | `03_原四步训练辨识与预测实现/` 保存四步训练与 Ridge 源程序，`04_原四步模型训练配置与记录/normalization.json` 为共同归一化与外推设置；`05_原四步模型及Ridge系数/` 保存权重和线性系数。 |
| 04 四步评估 | `06_原四步动态观测测试/scripts/python/evaluate_forecasts.py` 为原冻结推断入口，`results/prediction/cases/` 保存观测、辨识历史和逐点预测。 |
| 04 六步训练与评估 | `07_六步训练窗口与配置/train30.py` 为六步原实现，`08_六步模型及Ridge系数/` 提供权重和系数，`09_六步模型30ms逐点结果/` 包含原评分脚本及逐点互感输出。 |

移植时保留了原 NLS、复阻抗解码和探测、线性外推、网络类、四步窗口准备与预测、六步学习模型推断的函数体。标称电路与权重文件通过显式路径读取。六步的窗口准备和两种保持、Linear 的组合沿用 `evaluate30.py` 的索引、数据类型和算式。
