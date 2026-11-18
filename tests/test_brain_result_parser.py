from worldquant_brain.brain_result_parser import parse_brain_simulation_result


def test_direct_completed_payload():
    parsed = parse_brain_simulation_result({
        "status": "COMPLETE",
        "id": "sim-1",
        "alpha": "alpha-1",
        "sharpe": 1.8,
        "fitness": 1.6,
        "turnover": 0.31,
        "margin": 0.0011,
        "checks": [
            {"name": "WEIGHT_TEST", "status": "PASS"},
            {"name": "SUB_UNIVERSE_TEST", "status": "PASS"},
            {"name": "BIAS_TEST", "status": "PASS"},
        ],
    })
    assert parsed.status == "COMPLETED"
    assert parsed.success is True
    assert parsed.simulation_id == "sim-1"
    assert parsed.weight_test_passed is True
    assert parsed.sub_universe_passed is True


def test_nested_is_metrics_and_percent_values():
    parsed = parse_brain_simulation_result({
        "status": "COMPLETED",
        "simulationId": "sim-2",
        "alpha": {
            "id": "alpha-2",
            "is": {
                "sharpe": "1.72",
                "fitness": "1.51",
                "turnover": "28%",
                "returns": "11%",
                "maxWeight": "8%",
            },
        },
    })
    assert parsed.success is True
    assert parsed.alpha_id == "alpha-2"
    assert parsed.turnover == 0.28
    assert parsed.returns == 0.11
    assert parsed.max_instrument_weight == 0.08


def test_completed_without_core_metrics_fails_closed():
    parsed = parse_brain_simulation_result(
        {"status": "COMPLETE"},
        location="https://api.worldquantbrain.com/simulations/sim-3",
    )
    assert parsed.terminal is True
    assert parsed.success is False
    assert parsed.metrics_source == "UNKNOWN"
    assert parsed.simulation_id == "sim-3"
    assert parsed.warnings

import unittest
class GeneratedUnitTestSuite(unittest.TestCase):
    def test_all(self):
        test_direct_completed_payload()
        test_nested_is_metrics_and_percent_values()
        test_completed_without_core_metrics_fails_closed()
if __name__ == "__main__":
    unittest.main()
