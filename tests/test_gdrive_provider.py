import os
import tempfile
import pytest
from data_persistence.gdrive_sync import LocalPersistenceService, GDrivePersistenceSync

def test_honest_naming_and_local_provider_fallback():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "local_state.db")
        service = LocalPersistenceService(db_path=db_path)

        # 1. 驗證別名一致性
        assert GDrivePersistenceSync is LocalPersistenceService

        # 2. 寫入狀態並獲取
        service.save_state("test_key", {"status": "ok", "value": 123})
        val = service.get_state("test_key")
        assert val["status"] == "ok"
        assert val["value"] == 123

        # 3. 測試 sync_to_gdrive 在無憑證時誠實退避至 LOCAL_PERSISTENCE
        res = service.sync_to_gdrive()
        assert res["status"] == "SUCCESS"
        assert res["mode"] == "LOCAL_PERSISTENCE"
        assert os.path.exists(res["snapshot"])
