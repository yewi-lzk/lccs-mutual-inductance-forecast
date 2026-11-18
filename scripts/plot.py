"""Redraw the four numerical figures in English from the released CSVs."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path(__file__).resolve().parents[1] / "data"
CASES = [("id", "new_test_ID_constant_pass_00", "ID – constant pass"),
         ("ood", "new_test_OOD_chirp_00", "OOD – chirp")]
METHODS = [("GRU", "GRU", "o"), ("TCN_Residual", "Residual TCN", "v"),
           ("Ridge_Residual", "Ridge", "s"), ("Linear", "Linear", "^")]


def read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def column(records, key):
    return [float(row[key]) for row in records]


def electrical_signals():
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.0), constrained_layout=True)
    for i, (short, _, name) in enumerate(CASES):
        envelope = read(f"figures/fig04_{short}_envelope.csv")
        carrier = read(f"figures/fig04_{short}_carrier.csv")
        upper, lower = axes[:, i]
        upper.plot(column(envelope, "time_s"), [1000*v for v in column(envelope, "I_abs_A_RMS")])
        upper.set(title=f"{name} · 69 kHz", xlabel="Motion time (s)", ylabel="Primary current (mA RMS)")
        x = column(carrier, "window_time_us")
        voltage, = lower.plot(x, column(carrier, "Vs_V"), color="tab:blue", label="Drive voltage")
        right = lower.twinx()
        current, = right.plot(x, [1000*v for v in column(carrier, "Is_A")],
                              color="tab:orange", label="Input current")
        lower.set(xlabel="Time in final carrier window (µs)", ylabel="Voltage (V)")
        right.set_ylabel("Current (mA)")
        lower.legend(handles=[voltage, current], fontsize=8, loc="upper right")
    return fig


def current_estimation():
    rows = read("figures/fig06_current_nls.csv")
    time = column(rows, "time_s")
    fig, (current, mutual) = plt.subplots(1, 2, figsize=(9.3, 3.7), constrained_layout=True)
    for freq in (59, 64, 69, 74, 79):
        current.plot(time, column(rows, f"Is_{freq}kHz_magnitude_mA"), label=f"{freq} kHz")
    current.set(xlabel="Motion time (s)", ylabel="Primary current (mA RMS)", title="Five-frequency observation")
    current.legend(ncol=3, fontsize=8)
    mutual.plot(time, column(rows, "M_map_uH"), label="Map reference", color="black")
    mutual.plot(time, column(rows, "M_nls_uH"), label="NLS", color="tab:orange", alpha=.9)
    mutual.set(xlabel="Motion time (s)", ylabel="Mutual inductance (µH)", title="Same-time identification")
    mutual.legend()
    return fig


def forecasts():
    rows = read("figures/fig07_gru_forecast.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.7, 5.9), constrained_layout=True)
    for i, (_, case_id, name) in enumerate(CASES):
        upper, lower = axes[:, i]
        reference = {float(r["target_s"]): float(r["reference_M_uH"]) for r in rows if r["case_id"] == case_id}
        target_times = sorted(reference)
        upper.plot(target_times, [reference[t] for t in target_times], color="black", linestyle="--",
                   label="Map reference", zorder=3)
        for h in (10, 30):
            selected = sorted((r for r in rows if r["case_id"] == case_id and int(r["horizon_ms"]) == h),
                              key=lambda r: float(r["target_s"]))
            x = column(selected, "target_s")
            color = "tab:green" if h == 10 else "tab:orange"
            upper.plot(x, column(selected, "mean_GRU_M_uH"), color=color, label=f"GRU {h} ms")
            lower.plot(x, column(selected, "mean_individual_APE_pct"), color=color, label=f"{h} ms")
        upper.set(title=name, ylabel="Mutual inductance (µH)")
        lower.set(xlabel="Forecast target time (s)", ylabel="Mean individual APE (%)")
        upper.legend(fontsize=8)
    return fig


def sensitivity():
    horizons = read("tables/forecast_by_horizon.csv")
    noise_levels = read("tables/forecast_by_noise_at_10ms.csv")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2), constrained_layout=True)
    for i, domain in enumerate(("ID", "OOD")):
        left, right = axes[i]
        for method, name, marker in METHODS:
            by_h = {int(r["horizon_ms"]): float(r["M_MAPE_pct"]) for r in horizons
                    if r["domain"] == domain and r["method"] == method}
            xs = [5, 10, 15, 20]
            line, = left.plot(xs, [by_h[h] for h in xs], marker=marker, label=name)
            # 30 ms comes from a separately fitted six-step model, shown as an isolated point.
            left.plot(30, by_h[30], marker=marker, markerfacecolor="none", color=line.get_color())
            levels = sorted((r for r in noise_levels if r["domain"] == domain and r["method"] == method),
                            key=lambda r: float(r["noise_pct"]))
            right.plot(column(levels, "noise_pct"), column(levels, "M_MAPE_pct"), marker=marker,
                       color=line.get_color(), label=name)
        left.set(title=f"{domain} · 1% noise", xlabel="Forecast horizon (ms)", ylabel="Mutual-inductance MAPE (%)",
                 xticks=[5,10,15,20,30], ylim=(0,None))
        right.set(title=f"{domain} · 10 ms", xlabel="Phasor noise (%)", ylabel="Mutual-inductance MAPE (%)",
                  xticks=[0,.5,1,2], ylim=(0,None))
    axes[0,0].legend(fontsize=8)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/figures"),
                        help="directory for the redrawn PNGs (default: outputs/figures)")
    output = parser.parse_args().output
    output.mkdir(exist_ok=True, parents=True)
    plt.rcParams.update({"axes.grid": True, "grid.alpha": .2, "lines.linewidth": 1.2, "font.size": 9})
    for filename, plot in [("fig04-current", electrical_signals), ("fig06-nls", current_estimation),
                           ("fig07-forecast", forecasts), ("fig08-sensitivity", sensitivity)]:
        fig = plot()
        fig.savefig(output / f"{filename}.png", dpi=180)
        plt.close(fig)
        print(output / f"{filename}.png")


if __name__ == "__main__":
    main()
