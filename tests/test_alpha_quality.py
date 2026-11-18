from worldquant_brain.alpha_quality import AlphaMetrics, AlphaQuality, evaluate_alpha


def _metrics(**overrides):
    data = dict(alpha_id="a1", expression="rank(-returns)", delay=1, sharpe=1.8, fitness=1.6, returns=0.12, turnover=0.28, margin=0.0012, self_correlation=0.4, max_instrument_weight=0.07, sub_universe_passed=True, weight_test_passed=True, ranked_sharpe_passed=True, bias_test_passed=True, official_simulation_id="sim-1", metrics_source="WORLDQUANT_BRAIN")
    data.update(overrides)
    return AlphaMetrics(**data)


def test_official_good():
    report = evaluate_alpha(_metrics())
    assert report.quality == AlphaQuality.GOOD
    assert report.submit_ready is True
    assert report.good_ready is True


def test_local_proxy_never_good():
    report = evaluate_alpha(_metrics(metrics_source="LOCAL_PROXY", official_simulation_id=None, sharpe=3.0, fitness=3.0))
    assert report.quality == AlphaQuality.UNVERIFIED
    assert report.submit_ready is False
    assert report.good_ready is False


def test_high_turnover_is_submit_ready_but_not_good():
    report = evaluate_alpha(_metrics(turnover=0.68))
    assert report.submit_ready is True
    assert report.good_ready is False

import unittest
class GeneratedUnitTestSuite(unittest.TestCase):
    def test_all(self):
        test_official_good()
        test_local_proxy_never_good()
        test_high_turnover_is_submit_ready_but_not_good()
if __name__ == "__main__":
    unittest.main()
