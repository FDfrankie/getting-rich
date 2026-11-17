import os
import tempfile
import unittest
from unittest.mock import MagicMock

from worldquant_brain.alpha_vault import AlphaVault
from worldquant_brain.brain_job_queue import BrainJobQueue, BrainJobStatus


class TestPhase3Lifecycle(unittest.TestCase):
    def test_job_queue_vault_id_and_fail_callback(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            vault = AlphaVault(db_path=os.path.join(td, "v.db"))
            saved = vault.save_alpha("Bad_AI_Alpha", "nonexistent_op(close)", 1.5, 1.2, 0.2)
            vid = saved["alpha_id"]
            self.assertEqual(len(vault.list_alphas()), 1)

            def on_fail(job):
                if job.vault_id is not None:
                    vault.delete_alpha(job.vault_id)

            client = MagicMock()
            client.offline_mode = False
            client.authenticated = True
            client.BASE_URL = "https://api.worldquantbrain.com"
            client.timeout = 5.0
            client.session.post.side_effect = RuntimeError("Unknown operator nonexistent_op")

            q = BrainJobQueue(client, default_poll_interval=0.01, on_fail_callback=on_fail)
            job, _ = q.create_job("nonexistent_op(close)", vault_id=vid, auto_submit=True)

            import time
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline and job.status != BrainJobStatus.FAILED:
                time.sleep(0.02)

            self.assertEqual(job.status, BrainJobStatus.FAILED)
            self.assertEqual(len(vault.list_alphas()), 0)


if __name__ == "__main__":
    unittest.main()
