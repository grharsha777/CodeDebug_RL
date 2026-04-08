"""
Sandbox — Isolated execution directories for safe code evaluation.

Creates temporary working directories, writes agent-submitted code and
test files, and cleans up after execution. Designed so the isolation
layer can be upgraded to Docker/gVisor/Firecracker without changing
the executor interface.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger("codedebug.sandbox")


@dataclass
class SandboxConfig:
    """Configuration for the sandbox environment."""
    base_dir: str | None = None
    cleanup: bool = True
    max_file_size_bytes: int = 512 * 1024  # 512 KB per file
    blocked_imports: list[str] = field(
        default_factory=lambda: ["subprocess", "shutil", "socket", "http", "ctypes"]
    )
    allow_network: bool = False


class Sandbox:
    """
    Manages an isolated temporary directory for code execution.

    Usage:
        with Sandbox(config) as sb:
            sb.write_source("solution.py", code)
            sb.write_source("test_solution.py", test_code)
            # executor runs pytest inside sb.workdir
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()
        self._workdir: Path | None = None

    @property
    def workdir(self) -> Path:
        if self._workdir is None:
            raise RuntimeError("Sandbox not initialized — use as context manager")
        return self._workdir

    def __enter__(self) -> "Sandbox":
        base = self.config.base_dir or tempfile.gettempdir()
        self._workdir = Path(
            tempfile.mkdtemp(prefix="codedebug_", dir=base)
        )
        logger.debug("Sandbox created: %s", self._workdir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.config.cleanup and self._workdir and self._workdir.exists():
            shutil.rmtree(self._workdir, ignore_errors=True)
            logger.debug("Sandbox cleaned: %s", self._workdir)

    def write_source(self, filename: str, content: str) -> Path:
        """Write a source file into the sandbox directory."""
        if len(content.encode()) > self.config.max_file_size_bytes:
            raise ValueError(
                f"File {filename} exceeds max size "
                f"({self.config.max_file_size_bytes} bytes)"
            )
        filepath = self.workdir / filename
        filepath.write_text(content, encoding="utf-8")
        logger.debug("Wrote %s (%d bytes)", filename, len(content))
        return filepath

    def write_conftest(self) -> Path:
        """Write a minimal conftest.py to ensure clean pytest collection."""
        conftest = self.workdir / "conftest.py"
        conftest.write_text("# auto-generated conftest\n", encoding="utf-8")
        return conftest

    def get_file(self, filename: str) -> str:
        """Read a file from the sandbox."""
        filepath = self.workdir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"{filename} not in sandbox")
        return filepath.read_text(encoding="utf-8")

    def static_safety_check(self, code: str) -> list[str]:
        """
        Lightweight static analysis to flag obviously dangerous patterns.
        Returns a list of warnings (empty = OK). Not a security boundary —
        a first-pass filter that can be strengthened later.
        """
        warnings: list[str] = []
        for blocked in self.config.blocked_imports:
            if f"import {blocked}" in code or f"from {blocked}" in code:
                warnings.append(f"Blocked import detected: {blocked}")
        if "os.system(" in code or "exec(" in code or "eval(" in code:
            warnings.append("Potentially dangerous function call detected")
        if "__import__(" in code:
            warnings.append("Dynamic import detected")
        return warnings
