"""
Task Loader — Dataset loading and task sampling for CodeDebug-RL.

Supports:
1. Built-in starter tasks embedded in the package
2. On-disk task directories organized by difficulty
3. Deterministic sampling via seed
4. Future extension to HuggingFace datasets

Tasks are loaded as TaskSpec objects from JSON metadata + source files.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from codedebug_env.models import Difficulty, TaskSpec

logger = logging.getLogger("codedebug.task_loader")

# ─── Built-in Starter Tasks ──────────────────────────────────────────────────

BUILTIN_TASKS: list[TaskSpec] = [
    TaskSpec(
        task_id="builtin_001_fizzbuzz",
        difficulty=Difficulty.EASY,
        buggy_code="""\
def fizzbuzz(n: int) -> list[str]:
    \"\"\"Return FizzBuzz sequence from 1 to n.\"\"\"
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(i)  # BUG: should be str(i)
    return result
""",
        test_code="""\
from solution import fizzbuzz

def test_fizzbuzz_basic():
    result = fizzbuzz(15)
    assert isinstance(result, list)
    assert len(result) == 15

def test_fizzbuzz_types():
    result = fizzbuzz(5)
    assert all(isinstance(x, str) for x in result), "All elements must be strings"

def test_fizzbuzz_values():
    result = fizzbuzz(15)
    assert result[0] == "1"
    assert result[2] == "Fizz"
    assert result[4] == "Buzz"
    assert result[14] == "FizzBuzz"

def test_fizzbuzz_empty():
    assert fizzbuzz(0) == []
""",
        description="The fizzbuzz function should return a list of strings, but it returns integers for non-FizzBuzz numbers.",
        reference_solution="""\
def fizzbuzz(n: int) -> list[str]:
    \"\"\"Return FizzBuzz sequence from 1 to n.\"\"\"
    result = []
    for i in range(1, n + 1):
        if i % 3 == 0 and i % 5 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result
""",
        tags=["type-error", "easy", "strings"],
    ),
    TaskSpec(
        task_id="builtin_002_binary_search",
        difficulty=Difficulty.MEDIUM,
        buggy_code="""\
def binary_search(arr: list[int], target: int) -> int:
    \"\"\"Return index of target in sorted array, or -1 if not found.\"\"\"
    left, right = 0, len(arr)  # BUG: should be len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid  # BUG: should be mid - 1
    return -1
""",
        test_code="""\
from solution import binary_search

def test_found_middle():
    assert binary_search([1, 3, 5, 7, 9], 5) == 2

def test_found_first():
    assert binary_search([1, 3, 5, 7, 9], 1) == 0

def test_found_last():
    assert binary_search([1, 3, 5, 7, 9], 9) == 4

def test_not_found():
    assert binary_search([1, 3, 5, 7, 9], 4) == -1

def test_empty():
    assert binary_search([], 1) == -1

def test_single_element_found():
    assert binary_search([42], 42) == 0

def test_single_element_not_found():
    assert binary_search([42], 7) == -1
""",
        description="Binary search has off-by-one errors in boundary initialization and update.",
        reference_solution="""\
def binary_search(arr: list[int], target: int) -> int:
    \"\"\"Return index of target in sorted array, or -1 if not found.\"\"\"
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
""",
        tags=["off-by-one", "logic", "indexing", "medium"],
    ),
    TaskSpec(
        task_id="builtin_003_flatten_nested",
        difficulty=Difficulty.HARD,
        buggy_code="""\
def flatten(nested: list) -> list:
    \"\"\"Recursively flatten a nested list structure.\"\"\"
    result = []
    for item in nested:
        if isinstance(item, list):
            result.append(flatten(item))  # BUG: should be extend, not append
        else:
            result.append(item)
    return result

def sum_nested(nested: list) -> int:
    \"\"\"Sum all integers in a nested list.\"\"\"
    flat = flatten(nested)
    total = 0
    for x in flat:
        total += x  # will fail if flatten didn't actually flatten
    return total
""",
        test_code="""\
from solution import flatten, sum_nested

def test_flat_list():
    assert flatten([1, 2, 3]) == [1, 2, 3]

def test_one_level():
    assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]

def test_deep_nesting():
    assert flatten([1, [2, [3, [4]]]]) == [1, 2, 3, 4]

def test_empty():
    assert flatten([]) == []

def test_all_nested():
    assert flatten([[1], [2], [3]]) == [1, 2, 3]

def test_sum_nested_flat():
    assert sum_nested([1, 2, 3]) == 6

def test_sum_nested_deep():
    assert sum_nested([1, [2, [3]]]) == 6

def test_sum_nested_empty():
    assert sum_nested([]) == 0
""",
        description="flatten() uses append instead of extend for recursive results, producing nested rather than flat output. sum_nested depends on correct flatten behavior.",
        reference_solution="""\
def flatten(nested: list) -> list:
    \"\"\"Recursively flatten a nested list structure.\"\"\"
    result = []
    for item in nested:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

def sum_nested(nested: list) -> int:
    \"\"\"Sum all integers in a nested list.\"\"\"
    flat = flatten(nested)
    total = 0
    for x in flat:
        total += x
    return total
""",
        tags=["recursion", "logic", "hard", "data-structures"],
    ),
]


# ─── Task Loader ─────────────────────────────────────────────────────────────


class TaskLoader:
    """
    Loads and samples debugging tasks from built-in set or disk.

    Usage:
        loader = TaskLoader()
        loader.load_builtin()
        loader.load_directory("data/tasks")
        task = loader.sample(difficulty="easy", seed=42)
    """

    def __init__(self) -> None:
        self._tasks: dict[str, TaskSpec] = {}
        self._by_difficulty: dict[Difficulty, list[str]] = {
            d: [] for d in Difficulty
        }

    @property
    def task_count(self) -> int:
        return len(self._tasks)

    def load_builtin(self) -> int:
        """Load the built-in starter tasks. Returns count loaded."""
        for task in BUILTIN_TASKS:
            self._register(task)
        logger.info("Loaded %d built-in tasks", len(BUILTIN_TASKS))
        return len(BUILTIN_TASKS)

    def load_directory(self, path: str | Path) -> int:
        """
        Load tasks from a directory structure:
            path/
              easy/
                task_001/
                  source.py
                  test_source.py
                  metadata.json
        """
        root = Path(path)
        if not root.exists():
            logger.warning("Task directory not found: %s", root)
            return 0

        count = 0
        for difficulty_dir in root.iterdir():
            if not difficulty_dir.is_dir():
                continue
            difficulty_name = difficulty_dir.name.lower()
            try:
                difficulty = Difficulty(difficulty_name)
            except ValueError:
                # Not a difficulty folder — try loading individual tasks
                task = self._load_task_dir(difficulty_dir, Difficulty.MEDIUM)
                if task:
                    self._register(task)
                    count += 1
                continue

            for task_dir in difficulty_dir.iterdir():
                if not task_dir.is_dir():
                    continue
                task = self._load_task_dir(task_dir, difficulty)
                if task:
                    self._register(task)
                    count += 1

        logger.info("Loaded %d tasks from %s", count, root)
        return count

    def _load_task_dir(
        self, task_dir: Path, default_difficulty: Difficulty
    ) -> TaskSpec | None:
        """Load a single task from a directory."""
        source_file = task_dir / "source.py"
        test_file = task_dir / "test_source.py"
        meta_file = task_dir / "metadata.json"

        if not source_file.exists() or not test_file.exists():
            logger.debug("Skipping incomplete task dir: %s", task_dir)
            return None

        buggy_code = source_file.read_text(encoding="utf-8")
        test_code = test_file.read_text(encoding="utf-8")

        metadata: dict[str, Any] = {}
        if meta_file.exists():
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))

        # Load reference solution if present
        ref_file = task_dir / "reference.py"
        reference = ref_file.read_text(encoding="utf-8") if ref_file.exists() else None

        return TaskSpec(
            task_id=metadata.get("task_id", task_dir.name),
            difficulty=Difficulty(metadata.get("difficulty", default_difficulty.value)),
            buggy_code=buggy_code,
            canonical_filename=metadata.get("canonical_filename", "solution.py"),
            test_code=test_code,
            test_filename=metadata.get("test_filename", "test_solution.py"),
            description=metadata.get("description", ""),
            reference_solution=reference,
            tags=metadata.get("tags", []),
            metadata=metadata,
        )

    def _register(self, task: TaskSpec) -> None:
        """Register a task in the internal index."""
        self._tasks[task.task_id] = task
        if task.task_id not in self._by_difficulty[task.difficulty]:
            self._by_difficulty[task.difficulty].append(task.task_id)

    def get_task(self, task_id: str) -> TaskSpec | None:
        """Get a task by ID."""
        return self._tasks.get(task_id)

    def sample(
        self,
        difficulty: Difficulty | str | None = None,
        seed: int | None = None,
        exclude_ids: set[str] | None = None,
    ) -> TaskSpec:
        """
        Sample a random task, optionally filtered by difficulty.

        Args:
            difficulty: Filter to specific difficulty tier
            seed: Random seed for deterministic sampling
            exclude_ids: Task IDs to exclude (e.g., already seen)

        Returns:
            A TaskSpec
        """
        if not self._tasks:
            raise ValueError("No tasks loaded. Call load_builtin() or load_directory() first.")

        rng = random.Random(seed)
        exclude = exclude_ids or set()

        if difficulty is not None:
            if isinstance(difficulty, str):
                difficulty = Difficulty(difficulty)
            candidates = [
                tid
                for tid in self._by_difficulty[difficulty]
                if tid not in exclude
            ]
        else:
            candidates = [tid for tid in self._tasks if tid not in exclude]

        if not candidates:
            # Fallback to all tasks
            candidates = list(self._tasks.keys())

        task_id = rng.choice(candidates)
        return self._tasks[task_id]

    def list_tasks(self) -> list[dict[str, str]]:
        """List all available tasks with summary info."""
        return [
            {
                "task_id": t.task_id,
                "difficulty": t.difficulty.value,
                "description": t.description[:100],
                "tags": ", ".join(t.tags),
            }
            for t in self._tasks.values()
        ]

    def get_task_catalog(self) -> list[dict[str, Any]]:
        """Return rich task metadata for product-facing UIs."""
        return [
            {
                "task_id": task.task_id,
                "difficulty": task.difficulty.value,
                "description": task.description,
                "tags": task.tags,
                "has_reference_solution": task.reference_solution is not None,
                "test_filename": task.test_filename,
                "source_filename": task.canonical_filename,
                "metadata": task.metadata,
            }
            for task in sorted(
                self._tasks.values(),
                key=lambda task: (task.difficulty.value, task.task_id),
            )
        ]
