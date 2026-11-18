"""Compare one archived case against the published frozen inference modules."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .forecast import FourStep, SixStep
from .nls import CurrentEstimator


def compare(actual, archived):
    np.testing.assert_allclose(actual, archived, rtol=0, atol=2e-6)
    return float(np.max(np.abs(actual - archived)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive_root", type=Path, help="Parent of the extracted 01, 03 and 04 category directories")
    root = parser.parse_args().archive_root
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    em = root / "01_COMSOL_电磁建模" / "27位姿地图与原始端口记录" / "raw_impedance.csv"
    circuit = root / "03_MATLAB_Simulink_动态电路" / "标称补偿电路参数" / "configuration.json"
    research = root / "04_Python_PyTorch_辨识与预测"
    norm = research / "04_原四步模型训练配置与记录" / "normalization.json"
    four_dir = research / "05_原四步模型及Ridge系数"
    six_dir = research / "08_六步模型及Ridge系数"
    name = "new_test_OOD_chirp_00"
    first_record = research / "06_原四步动态观测测试/results/prediction/cases" / name / "n100_r0.npz"
    second_record = research / "09_六步模型30ms逐点结果/predictions" / f"{name}.npz"
    estimator = CurrentEstimator(em, circuit)
    short = FourStep(norm, four_dir / "learned_linear_baseline/coefficients.npz", four_dir / "weights")
    long = SixStep(norm, six_dir / "ridge30.npz", six_dir)
    with np.load(first_record, allow_pickle=False) as first, np.load(second_record, allow_pickle=False) as second:
        estimates = first["nls_theta_log"]
        one_packet = estimator.estimate(first["obs_noisy"][200:201])
        nls_delta = compare(one_packet, estimates[200:201])
        prepared = short.prepare(estimates)
        np.testing.assert_array_equal(prepared["ix"], first["origin_indices"])
        four = short.predict(prepared, estimates)
        ix, six = long.predict(estimates)
        np.testing.assert_array_equal(ix, second["origin_indices"])
        if len(four) != 10 or four.keys() != six.keys():
            raise ValueError("Expected four deterministic baselines and two networks with three seeds each")
        short_delta = max(compare(v, first[f"forecast_{k}_{s}"]) for (k, s), v in four.items())
        long_delta = max(compare(v[:, 0], second[f"forecast_M_log_{k}_{s}"][0]) for (k, s), v in six.items())
    print(f"Matched {name}, 1% observation noise, repeat 0.")
    print(f"NLS at t=1 s: maximum absolute log-parameter difference {nls_delta:.3g}")
    print(f"Four-step outputs, {len(prepared['ix'])} origins: maximum absolute log-parameter difference {short_delta:.3g}")
    print(f"Sixth-step M, {len(ix)} origins: maximum absolute log-M difference {long_delta:.3g}")


if __name__ == "__main__":
    main()
