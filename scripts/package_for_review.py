"""
Allowlist-based packaging script for review.
Generates an artifact bundle with SHA-256 MANIFEST.json while strictly
refusing .env, .git, *.db, and potential secrets.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple

FORBIDDEN_EXTENSIONS: Set[str] = {
    ".db",
    ".sqlite",
    ".sqlite3",
    ".joblib",
    ".pt",
    ".key",
    ".pem",
}

FORBIDDEN_NAMES: Set[str] = {
    ".env",
    "cloud_state_snapshot.json",
    "id_rsa",
    "id_ed25519",
}

FORBIDDEN_DIR_NAMES: Set[str] = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".hypothesis",
    "backups",
    "venv",
    ".venv",
}

SUSPICIOUS_PATTERNS: List[re.Pattern] = [
    re.compile(r"""(?i)(secret_key|api_secret|password|private_key)\s*=\s*['\"][A-Za-z0-9+/=_-]{16,}['\"]"""),
    re.compile(r"""-----BEGIN (?:EC|RSA|DSA|OPENSSH)?\s*PRIVATE KEY-----"""),
]


def sha256_of_file(file_path: Path) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def check_forbidden_rules(path: Path) -> None:
    parts = path.parts
    for p in parts:
        if p in FORBIDDEN_DIR_NAMES:
            raise ValueError(f"Packaging rejected: forbidden directory component '{p}' in {path}")
    name = path.name
    if name in FORBIDDEN_NAMES:
        raise ValueError(f"Packaging rejected: forbidden file '{name}' in {path}")
    if any(name.endswith(ext) for ext in FORBIDDEN_EXTENSIONS):
        raise ValueError(f"Packaging rejected: forbidden file extension for '{name}' in {path}")


def scan_for_secrets(file_path: Path) -> None:
    try:
        content = file_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return

    for pattern in SUSPICIOUS_PATTERNS:
        match = pattern.search(content)
        if match:
            raise ValueError(f"Packaging rejected: potential secret found in {file_path}: {match.group(0)[:20]}...")


def build_package(root_dir: Path, output_zip: Path, allowed_patterns: List[str] | None = None) -> Dict[str, str]:
    manifest: Dict[str, str] = {}
    root_dir = root_dir.resolve()

    if allowed_patterns is None:
        allowed_extensions = {".py", ".md", ".json", ".toml", ".txt", ".yaml", ".yml", ".example"}
    else:
        allowed_extensions = set(allowed_patterns)

    files_to_pack: List[Path] = []

    for curr_root, dirs, files in os.walk(root_dir):
        # Prune forbidden dirs in place
        dirs[:] = [d for d in dirs if d not in FORBIDDEN_DIR_NAMES and not d.startswith(".")]

        curr_p = Path(curr_root)
        for f in files:
            file_p = curr_p / f
            rel_p = file_p.relative_to(root_dir)

            try:
                check_forbidden_rules(rel_p)
            except ValueError:
                continue

            if file_p.suffix in allowed_extensions or file_p.name == ".env.example":
                check_forbidden_rules(rel_p)
                scan_for_secrets(file_p)
                files_to_pack.append(rel_p)

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel_p in sorted(files_to_pack):
            abs_p = root_dir / rel_p
            file_hash = sha256_of_file(abs_p)
            manifest[rel_p.as_posix()] = file_hash
            zf.write(abs_p, arcname=rel_p.as_posix())

        manifest_content = json.dumps(manifest, indent=2, sort_keys=True)
        zf.writestr("MANIFEST.json", manifest_content)

    return manifest


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    dist = root / "dist" / "review_package.zip"
    try:
        manifest = build_package(root, dist)
        print(f"Packaged {len(manifest)} files to {dist}")
    except Exception as e:
        print(f"Error packaging: {e}", file=sys.stderr)
        sys.exit(1)
