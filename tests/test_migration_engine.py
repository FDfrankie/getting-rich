"""Unit tests for the v5 transactional migration engine (dry-run, conflict, rollback, manifest recovery)."""

import json
import tempfile
import unittest
from pathlib import Path

import scripts.apply_full_audit_bundle_v5 as mig


class TestMigrationEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp_dir.name)
        self.manifest_path = self.root / ".patch_backup_manifest.json"
        self.backups_dir = self.root / ".patch_backups"

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_unknown_existing_file_causes_conflict(self):
        target = self.root / "crypto_trading" / "runtime_context.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# manually created file\n", encoding="utf-8")

        manifest = {"files": {}, "runs": {}}
        plan, has_err = mig.build_plan(manifest, base_root=self.root, subset_new={"crypto_trading/runtime_context.py": mig.RUNTIME_CONTEXT_PY}, subset_transforms={})
        self.assertTrue(has_err)
        self.assertEqual(plan["crypto_trading/runtime_context.py"][0], "CONFLICT")

    def test_apply_idempotency_manual_edit_conflict_and_rollback_manifest(self):
        # 1. Initial apply of a single new file + a transformed file
        env_file = self.root / ".env.example"
        env_file.write_text("BRAIN_EMAIL=\nBRAIN_API_KEY=\n", encoding="utf-8")

        rc = mig.execute_migration(
            apply_mode=True,
            base_root=self.root,
            manifest_path=self.manifest_path,
            backups_dir=self.backups_dir,
            subset_new={"crypto_trading/instruments.py": mig.INSTRUMENTS_PY},
            subset_transforms={".env.example": mig._transform_env_example},
            run_tests=False,
        )
        self.assertEqual(rc, 0)
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["runs"]), 1)
        run_id = next(iter(manifest["runs"].keys()))
        self.assertEqual(manifest["runs"][run_id]["status"], "APPLIED")

        # 2. Second apply is idempotent (ALREADY_APPLIED)
        plan2, err2 = mig.build_plan(
            manifest,
            base_root=self.root,
            subset_new={"crypto_trading/instruments.py": mig.INSTRUMENTS_PY},
            subset_transforms={".env.example": mig._transform_env_example},
        )
        self.assertFalse(err2)
        self.assertEqual(plan2["crypto_trading/instruments.py"][0], "ALREADY_APPLIED")
        self.assertEqual(plan2[".env.example"][0], "ALREADY_APPLIED")

        # 3. Manual modification triggers CONFLICT
        (self.root / "crypto_trading" / "instruments.py").write_text("# user edit\n", encoding="utf-8")
        plan3, err3 = mig.build_plan(
            manifest,
            base_root=self.root,
            subset_new={"crypto_trading/instruments.py": mig.INSTRUMENTS_PY},
            subset_transforms={".env.example": mig._transform_env_example},
        )
        self.assertTrue(err3)
        self.assertEqual(plan3["crypto_trading/instruments.py"][0], "CONFLICT")

        # 4. Rollback restores both files AND manifest entries
        rb_rc = mig.do_rollback(run_id, base_root=self.root, manifest_path=self.manifest_path)
        self.assertEqual(rb_rc, 0)
        self.assertFalse((self.root / "crypto_trading" / "instruments.py").exists())
        self.assertEqual(env_file.read_text(encoding="utf-8"), "BRAIN_EMAIL=\nBRAIN_API_KEY=\n")
        manifest_after = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest_after["runs"][run_id]["status"], "ROLLED_BACK")
        self.assertNotIn("crypto_trading/instruments.py", manifest_after["files"])


if __name__ == "__main__":
    unittest.main()
