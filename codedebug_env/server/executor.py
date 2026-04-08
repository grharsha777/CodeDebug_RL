"""
Executor — Safe code execution and pytest result parsing.

Handles:
1. Syntax validation via ast.parse / py_compile
2. Pytest execution in a subprocess with timeout
3. Structured parsing of test results
4. Stdout/stderr capture and truncation

Designed so the execution backend can be swapped (Docker, Firecracker)
without changing the interface.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from codedebug_env.models import ExecutionResult, ExecutionStatus, TestResult
from codedebug_env.server.sandbox import Sandbox, SandboxConfig

logger = logging.getLogger("codedebug.executor")

# Maximum output capture size to avoid memory issues
MAX_OUTPUT_BYTES = 64 * 1024  # 64 KB
DEFAULT_TIMEOUT_S = 30


def check_syntax(code: str) -> tuple[bool, str | None]:
    """
    Validate Python syntax using ast.parse.
    Returns (is_valid, error_message).
    """
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def run_pytest(
    workdir: Path,
    test_filename: str = "test_solution.py",
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> ExecutionResult:
    """
    Run pytest inside the given working directory and return structured results.

    Uses pytest's JSON report plugin if available, falling back to
    verbose output parsing.
    """
    start = time.monotonic()

    # Build pytest command with machine-readable output
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(workdir / test_filename),
        "-v",
        "--tb=short",
        "--no-header",
        "-q",
    ]

    # Try JSON report if plugin is available
    json_report_path = workdir / ".pytest_report.json"
    cmd_with_json = cmd + [
        "--json-report",
        f"--json-report-file={json_report_path}",
    ]

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONPATH"] = str(workdir)

    use_json = True
    try:
        result = subprocess.run(
            cmd_with_json,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(workdir),
            env=env,
        )
    except FileNotFoundError:
        # pytest not installed — shouldn't happen in our Docker image
        return ExecutionResult(
            status=ExecutionStatus.CRASH,
            syntax_valid=True,
            error_detail="pytest not found",
            duration_s=time.monotonic() - start,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            syntax_valid=True,
            stdout="",
            stderr="Execution timed out",
            duration_s=timeout_s,
        )

    duration = time.monotonic() - start

    # Try parsing JSON report first
    if json_report_path.exists():
        try:
            return _parse_json_report(json_report_path, result, duration)
        except Exception as e:
            logger.debug("JSON report parse failed: %s, falling back to stdout", e)
            use_json = False

    # If JSON report unavailable, re-run without that flag or parse stdout
    # Re-run without JSON report plugin
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(workdir),
            env=env,
        )
    except subprocess.TimeoutExpired:
        return ExecutionResult(
            status=ExecutionStatus.TIMEOUT,
            syntax_valid=True,
            duration_s=timeout_s,
        )
    duration = time.monotonic() - start
    return _parse_pytest_stdout(result, duration)


def _parse_json_report(
    report_path: Path,
    proc: subprocess.CompletedProcess[str],
    duration: float,
) -> ExecutionResult:
    """Parse pytest-json-report output for structured test results."""
    with open(report_path) as f:
        report = json.load(f)

    summary = report.get("summary", {})
    test_results: list[TestResult] = []

    for test in report.get("tests", []):
        nodeid = test.get("nodeid", "unknown")
        outcome = test.get("outcome", "unknown")
        test_duration = test.get("duration", 0.0)

        error_msg = None
        trace = None
        if outcome in ("failed", "error"):
            call_info = test.get("call", {})
            if call_info:
                crash = call_info.get("crash", {})
                error_msg = crash.get("message", "")
                trace = call_info.get("longrepr", "")
                if isinstance(trace, str) and len(trace) > 500:
                    trace = trace[:500] + "..."

        test_results.append(
            TestResult(
                name=nodeid,
                passed=outcome == "passed",
                duration_s=test_duration,
                error_message=error_msg,
                short_trace=trace,
            )
        )

    passed = summary.get("passed", 0)
    failed = summary.get("failed", 0)
    errored = summary.get("error", 0)
    skipped = summary.get("skipped", 0)
    total = summary.get("total", passed + failed + errored + skipped)

    status = ExecutionStatus.SUCCESS if failed == 0 and errored == 0 else ExecutionStatus.RUNTIME_ERROR

    return ExecutionResult(
        status=status,
        syntax_valid=True,
        total_tests=total,
        passed=passed,
        failed=failed,
        errored=errored,
        skipped=skipped,
        test_results=test_results,
        stdout=_truncate(proc.stdout),
        stderr=_truncate(proc.stderr),
        duration_s=duration,
    )


def _parse_pytest_stdout(
    proc: subprocess.CompletedProcess[str],
    duration: float,
) -> ExecutionResult:
    """
    Parse pytest verbose stdout output for test results.
    Handles the standard pytest -v output format.
    """
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    combined = stdout + "\n" + stderr

    # Check for collection errors or import errors
    if "ModuleNotFoundError" in combined or "ImportError" in combined:
        return ExecutionResult(
            status=ExecutionStatus.RUNTIME_ERROR,
            syntax_valid=True,
            stdout=_truncate(stdout),
            stderr=_truncate(stderr),
            duration_s=duration,
            error_detail="Import error during test collection",
        )

    test_results: list[TestResult] = []
    passed_count = 0
    failed_count = 0
    error_count = 0
    skipped_count = 0

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue

        # Parse verbose output: "test_file.py::test_name PASSED/FAILED"
        if " PASSED" in line:
            name = line.split(" PASSED")[0].strip()
            test_results.append(TestResult(name=name, passed=True))
            passed_count += 1
        elif " FAILED" in line:
            name = line.split(" FAILED")[0].strip()
            test_results.append(TestResult(name=name, passed=False))
            failed_count += 1
        elif " ERROR" in line:
            name = line.split(" ERROR")[0].strip()
            test_results.append(
                TestResult(name=name, passed=False, error_message="Error during test")
            )
            error_count += 1
        elif " SKIPPED" in line:
            skipped_count += 1

    # Parse summary line: "X passed, Y failed, Z error"
    for line in stdout.splitlines():
        if "passed" in line or "failed" in line or "error" in line:
            import re

            nums = re.findall(r"(\d+)\s+(passed|failed|error|skipped)", line)
            for count_str, kind in nums:
                count = int(count_str)
                if kind == "passed":
                    passed_count = max(passed_count, count)
                elif kind == "failed":
                    failed_count = max(failed_count, count)
                elif kind == "error":
                    error_count = max(error_count, count)
                elif kind == "skipped":
                    skipped_count = max(skipped_count, count)

    total = passed_count + failed_count + error_count + skipped_count
    all_pass = failed_count == 0 and error_count == 0 and total > 0
    status = ExecutionStatus.SUCCESS if all_pass else ExecutionStatus.RUNTIME_ERROR

    # Extract failure details from FAILURES section
    failure_traces = _extract_failure_traces(stdout)
    for tr in test_results:
        if not tr.passed and tr.name in failure_traces:
            tr.short_trace = failure_traces[tr.name]

    return ExecutionResult(
        status=status,
        syntax_valid=True,
        total_tests=total,
        passed=passed_count,
        failed=failed_count,
        errored=error_count,
        skipped=skipped_count,
        test_results=test_results,
        stdout=_truncate(stdout),
        stderr=_truncate(stderr),
        duration_s=duration,
    )


def _extract_failure_traces(stdout: str) -> dict[str, str]:
    """Extract short failure traces from pytest's FAILURES section."""
    traces: dict[str, str] = {}
    in_failures = False
    current_test = None
    current_trace: list[str] = []

    for line in stdout.splitlines():
        if "= FAILURES =" in line or "= ERRORS =" in line:
            in_failures = True
            continue
        if in_failures and line.startswith("_") and line.endswith("_"):
            # Save previous test trace
            if current_test and current_trace:
                trace_text = "\n".join(current_trace[-10:])  # last 10 lines
                traces[current_test] = trace_text[:500]
            # Parse test name from separator
            name = line.strip("_ ").strip()
            current_test = name
            current_trace = []
        elif in_failures and current_test:
            current_trace.append(line)
        if in_failures and ("= short test summary" in line or "==" in line and "passed" in line):
            if current_test and current_trace:
                trace_text = "\n".join(current_trace[-10:])
                traces[current_test] = trace_text[:500]
            break

    return traces


def execute_submission(
    source_code: str,
    test_code: str,
    source_filename: str = "solution.py",
    test_filename: str = "test_solution.py",
    timeout_s: float = DEFAULT_TIMEOUT_S,
    sandbox_config: SandboxConfig | None = None,
) -> ExecutionResult:
    """
    Full execution pipeline:
    1. Syntax check
    2. Write to sandbox
    3. Run pytest
    4. Parse and return structured results
    """
    # Step 1: Syntax validation
    valid, err = check_syntax(source_code)
    if not valid:
        return ExecutionResult(
            status=ExecutionStatus.SYNTAX_ERROR,
            syntax_valid=False,
            error_detail=err,
        )

    # Step 2: Execute in sandbox
    with Sandbox(sandbox_config) as sb:
        sb.write_source(source_filename, source_code)
        sb.write_source(test_filename, test_code)
        sb.write_conftest()

        # Step 3: Run pytest
        result = run_pytest(
            sb.workdir,
            test_filename=test_filename,
            timeout_s=timeout_s,
        )

    return result


def _truncate(text: str, max_bytes: int = MAX_OUTPUT_BYTES) -> str:
    """Truncate output to prevent memory issues."""
    if len(text.encode()) > max_bytes:
        return text[: max_bytes // 2] + "\n... [truncated] ...\n"
    return text
