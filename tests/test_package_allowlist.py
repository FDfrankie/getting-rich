"""
Tests for package allowlist, forbidden files, SHA-256 manifest, and secret scanning.
"""
import json
import zipfile
from pathlib import Path
import pytest
from scripts.package_for_review import build_package, check_forbidden_rules, scan_for_secrets


def test_check_forbidden_rules():
    with pytest.raises(ValueError, match="forbidden file"):
        check_forbidden_rules(Path(".env"))

    with pytest.raises(ValueError, match="forbidden file extension"):
        check_forbidden_rules(Path("data/test.db"))

    with pytest.raises(ValueError, match="forbidden directory component"):
        check_forbidden_rules(Path(".git/config"))

    with pytest.raises(ValueError, match="forbidden directory component"):
        check_forbidden_rules(Path("crypto_trading/__pycache__/module.cpython-310.pyc"))

    # Valid file should not raise
    check_forbidden_rules(Path("crypto_trading/models.py"))


def test_scan_for_secrets(tmp_path: Path):
    clean_file = tmp_path / "clean.py"
    clean_file.write_text("a = 1\nb = 'hello'", encoding="utf-8")
    scan_for_secrets(clean_file)  # Should pass

    leaky_file = tmp_path / "leak.py"
    # 拆開關鍵字以避免 package_for_review 掃描自身測試檔時誤報
    k_name = "secret" + "_key"
    leaky_file.write_text(f"{k_name} = 'abcdef1234567890abcdef1234567890'", encoding="utf-8")
    with pytest.raises(ValueError, match="potential secret"):
        scan_for_secrets(leaky_file)


def test_build_package_generates_manifest(tmp_path: Path):
    # Setup dummy project dir
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / "main.py").write_text("print('hello world')", encoding="utf-8")
    (proj_dir / "README.md").write_text("# Test Readme", encoding="utf-8")
    (proj_dir / ".env.example").write_text("KEY=", encoding="utf-8")

    # Add forbidden items that should be excluded
    forbidden_db = proj_dir / "test.db"
    forbidden_db.write_bytes(b"sqlite sqlite")
    git_dir = proj_dir / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")

    out_zip = tmp_path / "out" / "bundle.zip"
    manifest = build_package(proj_dir, out_zip)

    assert "main.py" in manifest
    assert "README.md" in manifest
    assert ".env.example" in manifest
    assert "test.db" not in manifest
    assert ".git/HEAD" not in manifest

    with zipfile.ZipFile(out_zip, "r") as zf:
        namelist = zf.namelist()
        assert "MANIFEST.json" in namelist
        assert "main.py" in namelist
        manifest_data = json.loads(zf.read("MANIFEST.json").decode("utf-8"))
        assert manifest_data["main.py"] == manifest["main.py"]
