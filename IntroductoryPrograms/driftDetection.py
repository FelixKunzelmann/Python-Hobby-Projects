import numpy as np
from scipy.stats import linregress


def detect_drift(timestamps: np.ndarray, values: np.ndarray, slope_threshold: float = 0.01) -> dict:

    slope, intercept, r_value, p_value, std_err = linregress(
        timestamps, values)

    is_drifting = abs(slope) > slope_threshold and p_value < 0.05

    return {
        "slope": slope,
        "p_value": p_value,
        "r_squared": r_value**2,
        "is_drifting": is_drifting,
        "verdict": "DRIFT DETECTED" if is_drifting else "STABLE"
    }


# Simulate drifting signal
t = np.linspace(0, 100, 200)
signal = 1.0 + 0.009 * t + np.random.normal(0, 0.1, len(t))

result = detect_drift(t, signal)
print(
    f"{result['verdict']} | slope={result['slope']:.4f} | p={result['p_value']:.4f}")
