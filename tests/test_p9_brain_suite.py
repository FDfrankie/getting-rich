"""
Mock Tests for P9 WorldQuant BRAIN Official Workflow and Auto-Submit.
8 tests required by docs/ROUND4_BRAIN_TAIWAN_GOVERNANCE_COMPACT.md:
1. test_brain_official_metrics.py
2. test_brain_restart_recovery.py
3. test_alpha_status_migration.py
4. test_brain_auto_submit_gate.py
5. test_brain_submission_idempotency.py
6. test_brain_submission_recovery.py
7. test_brain_submission_quota.py
8. test_brain_submission_kill_switch.py
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from worldquant_brain.brain_auto_submit import (
    KILL_SWITCH_ENV,
    REQUIRED_SUBMIT_ACK,
    SUBMIT_ACK_ENV,
    AlphaStatus,
    BrainAutoSubmitGate,
    migrate_alpha_status,
)
from worldquant_brain.brain_job_queue import BrainJobStatus, BrainSimulationJob


class TestBrainOfficialMetrics(unittest.TestCase):
    def test_official_metrics_validation(self):
        gate = BrainAutoSubmitGate()
        # Missing metric fails
        passed, reason = gate.evaluate_alpha({"sharpe": 1.5, "fitness": 1.2})
        self.assertFalse(passed)
        self.assertIn("Missing", reason)

        # Valid metrics pass
        valid = {"sharpe": 1.60, "fitness": 1.30, "margin": 0.0020, "turnover": 0.25}
        passed, reason = gate.evaluate_alpha(valid, correlation=0.2)
        self.assertTrue(passed)

        # Sub-par sharpe fails
        invalid = {"sharpe": 1.0, "fitness": 1.30, "margin": 0.0020, "turnover": 0.25}
        passed, reason = gate.evaluate_alpha(invalid)
        self.assertFalse(passed)


class TestBrainRestartRecovery(unittest.TestCase):
    def test_job_restoration(self):
        job = BrainSimulationJob("job_101", "key_hash_1", "close/open")
        job.simulation_location = "https://api.worldquantbrain.com/simulations/sim_abc"
        job.alpha_id = "alpha_xyz"
        job.status = BrainJobStatus.RUNNING

        # Simulate serialize and deserialize
        data = job.to_dict()
        recovered_job = BrainSimulationJob(data["job_id"], "key_hash_1", "close/open")
        recovered_job.simulation_location = data["simulation_location"]
        recovered_job.alpha_id = data["alpha_id"]
        recovered_job.status = BrainJobStatus(data["status"])

        self.assertEqual(recovered_job.job_id, "job_101")
        self.assertEqual(recovered_job.simulation_location, "https://api.worldquantbrain.com/simulations/sim_abc")
        self.assertEqual(recovered_job.alpha_id, "alpha_xyz")


class TestAlphaStatusMigration(unittest.TestCase):
    def test_status_migration(self):
        # Seed to DEMO_SAMPLE
        self.assertEqual(migrate_alpha_status({"status": "PASS", "is_seed": True}), AlphaStatus.DEMO_SAMPLE.value)

        # BTC proxy to LOCAL_SCREENED
        self.assertEqual(
            migrate_alpha_status({"status": "VERIFIED", "is_local_proxy": True}), AlphaStatus.LOCAL_SCREENED.value
        )

        # Missing alpha_id or location with VERIFIED to LEGACY_UNVERIFIED
        self.assertEqual(
            migrate_alpha_status({"status": "VERIFIED"}), AlphaStatus.LEGACY_UNVERIFIED.value
        )

        # With alpha_id to BRAIN_COMPLETE
        self.assertEqual(
            migrate_alpha_status({"status": "VERIFIED", "alpha_id": "alpha_123"}), AlphaStatus.BRAIN_COMPLETE.value
        )


class TestBrainAutoSubmitGate(unittest.TestCase):
    @patch.dict(os.environ, {SUBMIT_ACK_ENV: REQUIRED_SUBMIT_ACK, KILL_SWITCH_ENV: "0"})
    def test_auto_submit_gate(self):
        gate = BrainAutoSubmitGate()
        valid = {"sharpe": 1.60, "fitness": 1.30, "margin": 0.0020, "turnover": 0.25}
        can_submit, reason = gate.can_submit("key_test_1", valid, correlation=0.3)
        self.assertTrue(can_submit)

    @patch.dict(os.environ, {SUBMIT_ACK_ENV: "", KILL_SWITCH_ENV: "0"})
    def test_gate_fails_without_ack(self):
        gate = BrainAutoSubmitGate()
        valid = {"sharpe": 1.60, "fitness": 1.30, "margin": 0.0020, "turnover": 0.25}
        can_submit, reason = gate.can_submit("key_test_1", valid)
        self.assertFalse(can_submit)
        self.assertIn("ACK", reason)


class TestBrainSubmissionIdempotency(unittest.TestCase):
    @patch.dict(os.environ, {SUBMIT_ACK_ENV: REQUIRED_SUBMIT_ACK, KILL_SWITCH_ENV: "0"})
    def test_submission_idempotency(self):
        gate = BrainAutoSubmitGate(cooldown_seconds=0.0)
        valid = {"sharpe": 1.60, "fitness": 1.30, "margin": 0.0020, "turnover": 0.25}
        can_submit, _ = gate.can_submit("key_idem_1", valid)
        self.assertTrue(can_submit)

        gate.record_submission("alpha_1", "key_idem_1")

        # Second submission with same key is rejected
        can_submit2, reason2 = gate.can_submit("key_idem_1", valid)
        self.assertFalse(can_submit2)
        self.assertEqual(reason2, "ALREADY_SUBMITTED_IDEMPOTENT")


class TestBrainSubmissionRecovery(unittest.TestCase):
    def test_audit_log_recovery(self):
        gate = BrainAutoSubmitGate()
        gate.record_submission("alpha_rec", "key_rec", status="SUBMITTED")
        self.assertEqual(len(gate.audit_log), 1)
        self.assertEqual(gate.audit_log[0]["alpha_id"], "alpha_rec")


class TestBrainSubmissionQuota(unittest.TestCase):
    @patch.dict(os.environ, {SUBMIT_ACK_ENV: REQUIRED_SUBMIT_ACK, KILL_SWITCH_ENV: "0"})
    def test_quota_limit(self):
        gate = BrainAutoSubmitGate(daily_quota=2, cooldown_seconds=0.0)
        valid = {"sharpe": 1.60, "fitness": 1.30, "margin": 0.0020, "turnover": 0.25}

        # 1st
        c1, _ = gate.can_submit("k1", valid)
        self.assertTrue(c1)
        gate.record_submission("a1", "k1")

        # 2nd
        c2, _ = gate.can_submit("k2", valid)
        self.assertTrue(c2)
        gate.record_submission("a2", "k2")

        # 3rd should exceed
        c3, reason3 = gate.can_submit("k3", valid)
        self.assertFalse(c3)
        self.assertIn("QUOTA", reason3)


class TestBrainSubmissionKillSwitch(unittest.TestCase):
    @patch.dict(os.environ, {SUBMIT_ACK_ENV: REQUIRED_SUBMIT_ACK, KILL_SWITCH_ENV: "1"})
    def test_kill_switch(self):
        gate = BrainAutoSubmitGate()
        valid = {"sharpe": 1.60, "fitness": 1.30, "margin": 0.0020, "turnover": 0.25}
        can_submit, reason = gate.can_submit("k_kill", valid)
        self.assertFalse(can_submit)
        self.assertEqual(reason, "KILL_SWITCH_ACTIVE")


if __name__ == "__main__":
    unittest.main()
