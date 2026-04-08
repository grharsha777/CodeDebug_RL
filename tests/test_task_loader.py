"""Tests for the task loading module."""

import json
from pathlib import Path

import pytest

from codedebug_env.models import Difficulty, TaskSpec
from codedebug_env.server.task_loader import TaskLoader

def test_load_builtin_task():
    # Load an easy task
    loader = TaskLoader()
    loader.load_builtin()
    task1 = loader.get_task("builtin_001_fizzbuzz")
    assert isinstance(task1, TaskSpec)
    assert task1.task_id == "builtin_001_fizzbuzz"
    assert task1.difficulty == Difficulty.EASY
    assert "def " in task1.buggy_code
    assert "test_" in task1.test_code

    # Load a medium task
    task2 = loader.get_task("builtin_002_binary_search")
    assert task2.difficulty == Difficulty.MEDIUM


def test_load_nonexistent_task():
    loader = TaskLoader()
    loader.load_builtin()
    # get_task returns None if not found; the test previously expected ValueError
    assert loader.get_task("nonexistent_task") is None


def test_load_from_directory(tmp_path: Path):
    # Set up a structured task directory
    task_dir = tmp_path / "custom_task"
    task_dir.mkdir()

    metadata = {
        "task_id": "custom_task",
        "difficulty": "hard",
        "description": "Custom description",
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata))
    (task_dir / "source.py").write_text("def custom(): pass")
    (task_dir / "test_source.py").write_text("def test_custom(): pass")

    # Override OS environment to point to our temp directory for tasks
    import os

    os.environ["CODEDEBUG_TASK_DIR"] = str(tmp_path)
    try:
        loader = TaskLoader()
        loader.load_directory(tmp_path)
        task = loader.get_task("custom_task")
        assert task.task_id == "custom_task"
        assert task.difficulty == Difficulty.HARD
        assert task.buggy_code == "def custom(): pass"
        assert task.test_code == "def test_custom(): pass"
        assert task.description == "Custom description"
    finally:
        del os.environ["CODEDEBUG_TASK_DIR"]
