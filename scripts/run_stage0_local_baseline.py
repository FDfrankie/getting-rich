#!/usr/bin/env python3
"""Safe Stage 0 orchestrator for Antigravity Quant.

This script does not patch production code itself. It validates repository state,
audits the delta script, executes its true dry-run/apply modes, enforces an allowlist,
runs offline tests, stages exact files, and optionally creates a local commit.
It never pushes and never runs live tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

STAGE = "STAGE_0_LOCAL_BASELINE"

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_pytest_argv() -> list[str]:
    return [sys.executable, "-m", "pytest", "-m", LIVE_MARKERS, "-q"]


def _run_server_only(port: int = 18000) -> dict:
    return {
        "command_argv": [sys.executable, "-m", "web_ui.server"],
        "cwd": str(REPO_ROOT),
        "port": int(port),
        "server_module": "web_ui.server",
        "launched": False,
        "status": "NOT_EXECUTED",
    }


def run_tests(*, tests_only: bool = True) -> dict:
    if not tests_only:
        raise ValueError("Only tests-only mode is supported.")
    reports_dir = REPO_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    comp = subprocess.run([sys.executable, "-m", "compileall", "-q", "."], cwd=str(REPO_ROOT), shell=False, capture_output=True, text=True)
    ut = subprocess.run([sys.executable, "-m", "unittest", "tests.test_post_v5_delta", "tests.test_runtime_truth", "-v"], cwd=str(REPO_ROOT), shell=False, capture_output=True, text=True)
    pt = subprocess.run(build_pytest_argv(), cwd=str(REPO_ROOT), shell=False, capture_output=True, text=True)
    (reports_dir / "stage0_pytest_final.txt").write_text((pt.stdout or "") + (pt.stderr or ""), encoding="utf-8")
    passed = comp.returncode == 0 and ut.returncode == 0 and pt.returncode == 0
    summary = {
        "repository": str(REPO_ROOT),
        "python_executable": sys.executable,
        "compileall_exit_code": comp.returncode,
        "unittest_exit_code": ut.returncode,
        "pytest_exit_code": pt.returncode,
        "full_pytest_passed": pt.returncode == 0,
        "external_network_allowed": False,
        "bitunix_live_started": False,
        "brain_submit_executed": False,
        "final_status": "TESTS_PASSED" if passed else "TESTS_FAILED",
    }
    (reports_dir / "stage0_verification_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary

DEFAULT_BRANCH = "fix/post-v5-delta-v5"
DEFAULT_COMMIT = "fix: apply post-v5 delta v5 (Phase 0-5 baseline & fail-closed)"
FORBIDDEN_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pyc", ".zip", ".log", ".env"}
FORBIDDEN_PARTS = {"__pycache__", ".pytest_cache", "dist", ".git", ".post_v5_backups"}
LIVE_MARKERS = "not live and not brain_live and not binance_testnet_live and not bitunix_live"
PATCH_STATUS_RE = re.compile(r"^\s*-\s+(?P<path>.+?):\s+(?P<status>PATCHED|ALREADY_APPLIED|TARGET_NOT_FOUND|CONFLICT|FAILED)\s*$")

@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_seconds: float

@dataclass
class StageReport:
    stage: str = STAGE
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    base_branch: str | None = None
    branch: str | None = None
    base_commit: str | None = None
    new_commit: str | None = None
    push_performed: bool = False
    expected_targets: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    patch_statuses: dict[str, str] = field(default_factory=dict)
    commands: list[dict] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    final_status: str = "BLOCKED"

class StageError(RuntimeError):
    pass

def run(cmd: Sequence[str], cwd: Path, *, allow_failure: bool=False, env: dict[str,str] | None=None) -> CommandResult:
    import time
    start=time.monotonic()
    proc=subprocess.run(list(cmd), cwd=cwd, text=True, capture_output=True, env=env)
    result=CommandResult(list(cmd), proc.returncode, proc.stdout, proc.stderr, time.monotonic()-start)
    if proc.returncode and not allow_failure:
        raise StageError(f"Command failed ({proc.returncode}): {shlex.join(cmd)}\n{proc.stdout}\n{proc.stderr}")
    return result

def record(report: StageReport, result: CommandResult) -> None:
    report.commands.append({
        "command": result.command,
        "returncode": result.returncode,
        "stdout": result.stdout[-12000:],
        "stderr": result.stderr[-12000:],
        "duration_seconds": round(result.duration_seconds, 3),
    })

def git(repo: Path, *args: str, allow_failure: bool=False) -> CommandResult:
    return run(["git", *args], repo, allow_failure=allow_failure)

def ensure_repo(repo: Path) -> None:
    if not (repo/'.git').exists():
        r=git(repo, "rev-parse", "--is-inside-work-tree", allow_failure=True)
        if r.returncode or r.stdout.strip() != "true":
            raise StageError(f"Not a Git repository: {repo}")

def ensure_clean(repo: Path, report: StageReport) -> None:
    r=git(repo, "status", "--porcelain=v1")
    record(report,r)
    if r.stdout.strip():
        raise StageError("Working tree is not clean. Refusing to stash, reset, or mix existing changes.")

def current_branch(repo: Path) -> str:
    return git(repo, "branch", "--show-current").stdout.strip()

def switch_branch(repo: Path, branch: str, report: StageReport) -> None:
    if current_branch(repo) != "main":
        return
    exists=git(repo, "show-ref", "--verify", f"refs/heads/{branch}", allow_failure=True)
    cmd=("switch", branch) if exists.returncode==0 else ("switch", "-c", branch)
    r=git(repo, *cmd); record(report,r)

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1<<20), b''): h.update(chunk)
    return h.hexdigest()

def extract_targets(script_text: str) -> list[str]:
    targets=set()
    # Explicit transform/new-file dictionary keys.
    for m in re.finditer(r"^[ \t]*[\"']([^\"']+\.(?:py|js|html|md|json|toml|yaml|yml))[\"']\s*:\s*", script_text, re.M):
        targets.add(m.group(1))
    # Conservative ROOT / "path" detection.
    for m in re.finditer(r"ROOT\s*/\s*[\"']([^\"']+)[\"']", script_text):
        rel=m.group(1)
        if Path(rel).suffix: targets.add(rel)
    return sorted(targets)

def audit_patch_script(repo: Path, script_rel: str, report: StageReport) -> Path:
    path=repo/script_rel
    if not path.exists():
        raise StageError(f"Required patch script is missing: {script_rel}. Do not copy or rename a similar version.")
    txt=path.read_text(encoding='utf-8')
    report.expected_targets=extract_targets(txt)
    if "--apply" not in txt:
        raise StageError("Patch script does not expose an explicit --apply mode.")
    if "PLAN_DRY_RUN" not in txt and "dry" not in txt.lower():
        report.warnings.append("Dry-run marker not obvious; bare execution will be treated as plan only and verified by Git diff.")
    if re.search(r"\b(push|git push)\b", txt, re.I):
        raise StageError("Patch script contains a push operation or push-like command.")
    scope={
        "script": script_rel,
        "script_sha256": sha256_file(path),
        "base_commit": report.base_commit,
        "expected_targets": report.expected_targets,
        "may_delete": bool(re.search(r"unlink\(|rmtree\(", txt)),
        "uses_subprocess": "subprocess" in txt,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    out=repo/'reports/post_v5_delta_v5_scope.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scope,indent=2,ensure_ascii=False),encoding='utf-8')
    if 'reports/post_v5_delta_v5_scope.json' not in report.expected_targets:
        report.expected_targets.append('reports/post_v5_delta_v5_scope.json')
    return path

def parse_patch_statuses(text: str) -> dict[str,str]:
    statuses={}
    for line in text.splitlines():
        m=PATCH_STATUS_RE.match(line)
        if m: statuses[m.group('path')]=m.group('status')
    return statuses

def changed_files(repo: Path) -> list[str]:
    r=git(repo,"status","--porcelain=v1","-uall")
    files=[]
    for line in r.stdout.splitlines():
        if not line.strip(): continue
        raw=line[3:]
        if ' -> ' in raw: raw=raw.split(' -> ',1)[1]
        clean = raw.strip('"')
        if clean.startswith(".post_v5_backups/"):
            continue
        files.append(clean)
    return sorted(set(files))

def assert_scope(files: Iterable[str], allowlist: set[str]) -> None:
    extra=sorted(set(files)-allowlist)
    if extra:
        raise StageError(f"Out-of-scope changes detected: {extra}")

def forbidden_path(path: str) -> bool:
    p=Path(path)
    return p.name=='.env' or p.suffix.lower() in FORBIDDEN_SUFFIXES or any(part in FORBIDDEN_PARTS for part in p.parts)

def audit_changed_content(repo: Path, files: Iterable[str]) -> list[str]:
    findings=[]
    patterns={
        "HARDCODED_64500": re.compile(r"(?<![\d.])64500(?:\.0+)?(?![\d.])"),
        "HARDCODED_15000": re.compile(r"(?<![\d.])15000(?:\.0+)?(?![\d.])"),
        "HARDCODED_112_50": re.compile(r"(?<![\d.])112\.50(?![\d.])"),
        "TEST_SKIP": re.compile(r"pytest\.skip|unittest\.skip|@pytest\.mark\.skip|xfail",re.I),
        "WEAK_ASSERT": re.compile(r"assert\s+True\b|self\.assertTrue\(True\)"),
    }
    for rel in files:
        if forbidden_path(rel): findings.append(f"FORBIDDEN_ARTIFACT:{rel}"); continue
        p=repo/rel
        if not p.exists() or p.is_dir() or p.suffix.lower() not in {'.py','.js','.html','.md','.toml','.json'}: continue
        is_test_or_patcher = rel.startswith('tests/') or rel.startswith('scripts/apply_post_v5_delta') or rel.startswith('reports/')
        diff_res = git(repo, 'diff', '--unified=0', '--', rel, allow_failure=True)
        added_lines = "\n".join(
            ln[1:] for ln in diff_res.stdout.splitlines() if ln.startswith('+') and not ln.startswith('+++')
        )
        txt = added_lines if added_lines.strip() else p.read_text(encoding='utf-8', errors='replace')
        for name, rx in patterns.items():
            if name.startswith('HARDCODED_') and is_test_or_patcher:
                continue
            if rx.search(txt): findings.append(f"{name}:{rel}")
    return findings

def run_test(report: StageReport, repo: Path, cmd: list[str], *, required=True) -> None:
    r=run(cmd,repo,allow_failure=True); record(report,r)
    if required and r.returncode:
        raise StageError(f"Required verification failed: {shlex.join(cmd)}")

def precise_stage(repo: Path, files: list[str], report: StageReport) -> None:
    safe=[f for f in files if not forbidden_path(f)]
    if set(safe)!=set(files):
        raise StageError("Forbidden artifacts are present; refusing to stage.")
    if not safe: raise StageError("No files to stage.")
    r=git(repo,"add","--",*safe); record(report,r)
    for args in [("diff","--cached","--check"),("diff","--cached","--name-status"),("diff","--cached","--stat")]:
        rr=git(repo,*args); record(report,rr)

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,default=Path.cwd())
    ap.add_argument('--patch-script',default='scripts/apply_post_v5_delta_v5.py')
    ap.add_argument('--branch',default=DEFAULT_BRANCH)
    ap.add_argument('--apply',action='store_true',help='Apply patch after successful plan')
    ap.add_argument('--commit',action='store_true',help='Create a local commit after all checks')
    ap.add_argument('--commit-message',default=DEFAULT_COMMIT)
    ap.add_argument('--allow-target',action='append',default=[],help='Additional allowed changed path')
    ap.add_argument('--tests-only',action='store_true',help='Run compileall, unittest, and offline pytest only')
    ap.add_argument('--server-only',action='store_true',help='Report server metadata without launching stub')
    ap.add_argument('--port',type=int,default=18000,help='Server port')
    args=ap.parse_args()
    if args.tests_only:
        s = run_tests(tests_only=True)
        print(json.dumps(s, ensure_ascii=False))
        return 0 if s['final_status'] == 'TESTS_PASSED' else 1
    if args.server_only:
        print(json.dumps(_run_server_only(args.port), ensure_ascii=False))
        return 1
    repo=args.repo.resolve(); report=StageReport()
    report_path=repo/'reports/stage0_execution_report.json'
    try:
        ensure_repo(repo)
        report.base_branch=current_branch(repo)
        report.base_commit=git(repo,'rev-parse','HEAD').stdout.strip()
        ensure_clean(repo,report)
        switch_branch(repo,args.branch,report)
        report.branch=current_branch(repo)
        if report.branch=='main': raise StageError('Refusing to operate on main.')
        script=audit_patch_script(repo,args.patch_script,report)

        # Script audit creates the scope file; treat it as the sole expected pre-plan change.
        pre_plan=changed_files(repo)
        assert_scope(pre_plan, {'reports/post_v5_delta_v5_scope.json'})

        plan=run([sys.executable,str(script)],repo,allow_failure=True); record(report,plan)
        report.patch_statuses=parse_patch_statuses(plan.stdout+'\n'+plan.stderr)
        bad={p:s for p,s in report.patch_statuses.items() if s in {'TARGET_NOT_FOUND','CONFLICT','FAILED'}}
        if plan.returncode or bad:
            raise StageError(f"Patch plan blocked: {bad or plan.stderr[-1000:]}")
        if not args.apply:
            report.final_status='PARTIAL'
            report.warnings.append('Dry-run/plan completed. Re-run with --apply after review.')
            return 0

        applied=run([sys.executable,str(script),'--apply'],repo,allow_failure=True); record(report,applied)
        if applied.returncode: raise StageError('Patch apply failed.')
        report.patch_statuses=parse_patch_statuses(applied.stdout+'\n'+applied.stderr) or report.patch_statuses
        files=changed_files(repo); report.changed_files=files
        allow=set(report.expected_targets)|set(args.allow_target)|{args.patch_script,'reports/stage0_execution_report.json','reports/post_v5_delta_v5_scope.json'}
        assert_scope(files,allow)
        findings=audit_changed_content(repo,files)
        if findings: raise StageError('Safety audit findings: '+', '.join(findings))

        run_test(report,repo,[sys.executable,'-m','compileall','-q','.'])
        run_test(report,repo,[sys.executable,'-m','pytest','tests/test_post_v5_delta.py','-q'])
        run_test(report,repo,[sys.executable,'-m','pytest','tests/test_runtime_truth.py','-q'])
        run_test(report,repo,[sys.executable,'-m','pytest','-m',LIVE_MARKERS,'-q'])
        # Ruff is required only when installed.
        probe=run(['ruff','--version'],repo,allow_failure=True); record(report,probe)
        if probe.returncode==0:
            py_targets = [f for f in changed_files(repo) if f.endswith('.py') and (repo / f).exists()]
            if py_targets:
                run_test(report,repo,['ruff','check',*py_targets])
        else:
            report.warnings.append('Ruff NOT RUN: executable unavailable.')

        report_path.parent.mkdir(parents=True,exist_ok=True)
        report.final_status='PARTIAL'
        report_path.write_text(json.dumps(asdict(report),indent=2,ensure_ascii=False),encoding='utf-8')
        files=changed_files(repo); report.changed_files=files
        assert_scope(files,allow)
        findings=audit_changed_content(repo,files)
        if findings: raise StageError('Pre-stage safety findings: '+', '.join(findings))

        if args.commit:
            precise_stage(repo,files,report)
            r=git(repo,'commit','-m',args.commit_message); record(report,r)
            report.new_commit=git(repo,'rev-parse','HEAD').stdout.strip()
            clean=git(repo,'status','--porcelain=v1'); record(report,clean)
            if clean.stdout.strip(): raise StageError('Commit created but working tree is not clean.')
            report.final_status='LOCAL_BASELINE_COMPLETE'
        else:
            report.warnings.append('Changes verified but not committed. Re-run with --commit after diff review.')
            report.final_status='PARTIAL'
        return 0
    except StageError as exc:
        report.blockers.append(str(exc)); report.final_status='BLOCKED'
        return 1
    finally:
        report.branch=report.branch or (current_branch(repo) if (repo/'.git').exists() else None)
        report_path.parent.mkdir(parents=True,exist_ok=True)
        report_path.write_text(json.dumps(asdict(report),indent=2,ensure_ascii=False),encoding='utf-8')
        print(json.dumps({"stage":report.stage,"status":report.final_status,"report":str(report_path)},ensure_ascii=False))

if __name__=='__main__':
    raise SystemExit(main())
