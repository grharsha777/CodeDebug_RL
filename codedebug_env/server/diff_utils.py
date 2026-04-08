"""
Diff Utilities — Human-readable patch diffs and change metrics.

Provides unified diff generation, diff statistics for reward computation,
and patch application support. Used by the reward system to measure
patch efficiency and by observations to show what changed.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass


@dataclass
class DiffStats:
    """Quantitative summary of a code diff."""
    lines_added: int = 0
    lines_removed: int = 0
    lines_changed: int = 0
    hunks: int = 0
    total_diff_lines: int = 0
    is_noop: bool = True

    @property
    def churn(self) -> int:
        """Total line churn — a proxy for patch size."""
        return self.lines_added + self.lines_removed


def compute_unified_diff(
    old_code: str,
    new_code: str,
    filename: str = "solution.py",
    context_lines: int = 3,
) -> str:
    """
    Generate a unified diff string between old and new code.
    Returns an empty string if codes are identical.
    """
    old_lines = old_code.splitlines(keepends=True)
    new_lines = new_code.splitlines(keepends=True)
    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{filename}",
        tofile=f"b/{filename}",
        n=context_lines,
    )
    return "".join(diff)


def compute_diff_stats(old_code: str, new_code: str) -> DiffStats:
    """
    Compute quantitative diff statistics for reward computation.
    Uses SequenceMatcher for accurate change detection.
    """
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()

    if old_lines == new_lines:
        return DiffStats(is_noop=True)

    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    stats = DiffStats(is_noop=False)
    in_hunk = False

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            in_hunk = False
            continue

        if not in_hunk:
            stats.hunks += 1
            in_hunk = True

        if tag == "replace":
            stats.lines_changed += max(i2 - i1, j2 - j1)
            stats.lines_removed += i2 - i1
            stats.lines_added += j2 - j1
        elif tag == "delete":
            stats.lines_removed += i2 - i1
        elif tag == "insert":
            stats.lines_added += j2 - j1

    stats.total_diff_lines = stats.lines_added + stats.lines_removed
    return stats


def apply_unified_diff(original: str, diff_text: str) -> str:
    """
    Apply a unified diff to original source code.

    This is a best-effort implementation for the unified_diff patch format.
    Falls back to returning original on parse errors (with a logged warning).
    """
    import re
    import logging

    logger = logging.getLogger("codedebug.diff_utils")

    lines = original.splitlines(keepends=True)
    result: list[str] = []
    diff_lines = diff_text.splitlines(keepends=True)

    hunk_pattern = re.compile(r"^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    idx = 0
    src_line = 0

    # Skip header lines
    while idx < len(diff_lines) and not diff_lines[idx].startswith("@@"):
        idx += 1

    try:
        while idx < len(diff_lines):
            m = hunk_pattern.match(diff_lines[idx])
            if not m:
                idx += 1
                continue

            hunk_start = int(m.group(1)) - 1  # 0-indexed
            # Copy lines before this hunk
            while src_line < hunk_start and src_line < len(lines):
                result.append(lines[src_line])
                src_line += 1

            idx += 1
            while idx < len(diff_lines) and not diff_lines[idx].startswith("@@"):
                line = diff_lines[idx]
                if line.startswith("-"):
                    src_line += 1  # skip removed line
                elif line.startswith("+"):
                    result.append(line[1:])  # add new line
                elif line.startswith(" "):
                    result.append(lines[src_line] if src_line < len(lines) else line[1:])
                    src_line += 1
                else:
                    # context or junk line
                    pass
                idx += 1

        # Copy remaining lines
        while src_line < len(lines):
            result.append(lines[src_line])
            src_line += 1

        return "".join(result)

    except (IndexError, ValueError) as e:
        logger.warning("Failed to apply unified diff: %s. Returning original.", e)
        return original


def truncate_diff(diff_text: str, max_lines: int = 50) -> str:
    """Truncate a diff to a maximum number of lines for display."""
    lines = diff_text.splitlines()
    if len(lines) <= max_lines:
        return diff_text
    return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"
