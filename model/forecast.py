"""Frozen four- and six-step inference from standardized log-parameter histories."""
import json
from pathlib import Path

import numpy as np
import torch

from . import nls as phys
from .networks import Forecast, Forecast30


SEEDS = (20261110, 20261111, 20261112)


def load_models(network, directory):
    result = {}
    for name in ("GRU", "TCN_Residual"):
        for seed in SEEDS:
            model = network(name)
            weights = torch.load(directory / f"{name}_{seed}.pt", map_location="cpu", weights_only=True)
            model.load_state_dict(weights, strict=True)
            result[name, seed] = model.eval()
    return result


class FourStep:
    def __init__(self, normalization, ridge, weights):
        config = json.loads(Path(normalization).read_text(encoding="utf-8-sig"))
        self.mean = np.asarray(config["mean"])
        self.std = np.asarray(config["std"])
        self.linear_cfg = config["linear"]
        with np.load(ridge, allow_pickle=False) as fit:
            np.testing.assert_array_equal(fit["normalization_mean"], self.mean)
            np.testing.assert_array_equal(fit["normalization_std"], self.std)
            self.coef = fit["coefficient"]
            self.intercept = fit["intercept"]
        if self.coef.shape != (96, 12) or self.intercept.shape != (12,):
            raise ValueError("Expected four-step Ridge coefficients")
        self.models = load_models(Forecast, Path(weights))

    def prepare(self, estimates):
        level, linear = phys.linear_prediction(estimates, self.linear_cfg["window"], self.linear_cfg["decay"])
        ix = np.arange(31, len(estimates)-4)
        histories = np.stack([estimates[i-31:i+1] for i in ix])
        return dict(ix=ix, x=((histories-self.mean)/self.std).transpose(0, 2, 1).astype("float32"),
                    level=level[ix].astype("float32"), linear=linear[ix].astype("float32"))

    def predict(self, prepared, estimates=None):
        p = prepared
        output = {("Linear", 0): np.clip(p["linear"], phys.LOW, phys.HIGH)}
        correction = (p["x"].reshape(-1, 96).astype("float64") @ self.coef + self.intercept).reshape(-1, 4, 3)
        output[("Ridge_Residual", 0)] = np.clip(p["linear"] + correction*self.std, phys.LOW, phys.HIGH)
        output[("DenoisedHold", 0)] = np.clip(np.repeat(p["level"][:, None, :], 4, axis=1), phys.LOW, phys.HIGH)
        if estimates is not None:
            output[("CurrentHold", 0)] = np.clip(np.repeat(estimates[p["ix"], None, :], 4, axis=1), phys.LOW, phys.HIGH)
        with torch.inference_mode():
            x = torch.from_numpy(p["x"])
            for (method, seed), model in self.models.items():
                base = p["linear"] if method == "TCN_Residual" else np.repeat(p["level"][:, None, :], 4, axis=1)
                output[(method, seed)] = np.clip(base + model(x).numpy()*self.std, phys.LOW, phys.HIGH)
        return output


class SixStep:
    def __init__(self, normalization, ridge, weights):
        config = json.loads(Path(normalization).read_text(encoding="utf-8-sig"))
        self.mean = np.asarray(config["mean"])
        self.std = np.asarray(config["std"])
        self.linear_cfg = config["linear"]
        with np.load(ridge, allow_pickle=False) as fit:
            self.coef = fit["coef"]
            self.intercept = fit["intercept"]
        if self.coef.shape != (96, 18) or self.intercept.shape != (18,):
            raise ValueError("Expected six-step Ridge coefficients")
        self.models = load_models(Forecast30, Path(weights))

    def predict(self, estimates):
        """Return origin indices and predictions at the sixth step, in log coordinates."""
        ix = np.arange(31, len(estimates) - 6)
        level, short_linear = phys.linear_prediction(
            estimates, self.linear_cfg["window"], self.linear_cfg["decay"]
        )
        history = estimates[ix[:, None] + np.arange(-31, 1)]
        x = ((history - self.mean) / self.std).transpose(0, 2, 1).astype(np.float32)
        slope = short_linear[ix, 0] - level[ix]
        linear = (level[ix] + 6 * slope).astype(np.float32)
        current = estimates[ix]
        level = level[ix].astype(np.float32)
        result = {
            (method, 0): np.clip(values, phys.LOW, phys.HIGH)
            for method, values in [("CurrentHold", current), ("DenoisedHold", level), ("Linear", linear)]
        }
        result.update(self.predict30(x, level, linear))
        return ix, result

    def predict30(self, x, level, linear30):
        """Predict absolute clipped log states from normalized history and causal bases."""
        x = np.asarray(x, dtype="float32")
        assert x.shape[1:] == (3, 32) and level.shape == linear30.shape == (len(x), 3)
        correction = x.reshape(-1, 96).astype("float64") @ self.coef[:, -3:] + self.intercept[-3:]
        output = {("Ridge_Residual", 0): np.clip(linear30 + correction * self.std, phys.LOW, phys.HIGH)}
        with torch.inference_mode():
            xt = torch.from_numpy(x)
            for (name, seed), model in self.models.items():
                base = linear30 if name == "TCN_Residual" else level
                correction = model(xt).numpy()[:, -1]
                output[name, seed] = np.clip(base + correction * self.std, phys.LOW, phys.HIGH)
        return output
