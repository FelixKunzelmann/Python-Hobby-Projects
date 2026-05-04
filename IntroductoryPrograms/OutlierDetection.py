import pytest
import numpy as np


def measure_output_power(channel: int) -> float:
    # Simulates reading from an instrument
    return np.random.normal(loc=-10.0, scale=4)  # dBm


POWER_SPEC = {
    "min_dBm": -11.0,
    "max_dBm": -9.0
}


def test_output_power():
    NUM_SAMPLES = 6
    results = [measure_output_power(channel=1) for _ in range(NUM_SAMPLES)]
    mean_power = np.mean(results)

    assert POWER_SPEC["min_dBm"] <= mean_power <= POWER_SPEC["max_dBm"], (
        f"Output power {mean_power:.2f} dBm is out of spec "
        f"[{POWER_SPEC['min_dBm']}, {POWER_SPEC['max_dBm']}] dBm"
    )
    print(f"PASS: Mean output power = {mean_power:.2f} dBm")
