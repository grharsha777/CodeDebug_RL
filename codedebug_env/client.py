"""
CodeDebug-RL Client — OpenEnv-compatible HTTP client for the environment.

Provides a clean Python API for interacting with the CodeDebug-RL
environment server. Handles serialization, deserialization, and
error handling for all OpenEnv endpoints.

Usage:
    from codedebug_env.client import CodeDebugClient

    client = CodeDebugClient("http://localhost:8000")
    obs = client.reset(difficulty="easy")
    obs, reward, done, info = client.step(
        CodeDebugAction(patched_code="...", reasoning="...")
    )
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from codedebug_env.models import CodeDebugAction, CodeDebugObservation

logger = logging.getLogger("codedebug.client")

DEFAULT_TIMEOUT = 60.0


class CodeDebugClient:
    """
    HTTP client for the CodeDebug-RL environment server.

    Wraps all OpenEnv endpoints with typed Python methods.
    Compatible with synchronous training loops (GRPO, PPO, etc.).
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> "CodeDebugClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    # ─── OpenEnv Endpoints ────────────────────────────────────────────────

    def health(self) -> dict[str, str]:
        """Check environment health."""
        resp = self._client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def reset(
        self,
        task_id: str | None = None,
        difficulty: str | None = None,
        seed: int | None = None,
    ) -> CodeDebugObservation:
        """
        Reset the environment and start a new episode.

        Args:
            task_id: Specific task to load
            difficulty: Filter by difficulty ("easy", "medium", "hard")
            seed: Random seed for deterministic selection

        Returns:
            Initial CodeDebugObservation
        """
        payload: dict[str, Any] = {}
        if task_id:
            payload["task_id"] = task_id
        if difficulty:
            payload["difficulty"] = difficulty
        if seed is not None:
            payload["seed"] = seed

        resp = self._client.post("/reset", json=payload)
        resp.raise_for_status()
        data = resp.json()
        return CodeDebugObservation(**data["observation"])

    def step(
        self, action: CodeDebugAction
    ) -> tuple[CodeDebugObservation, float, bool, dict[str, Any]]:
        """
        Submit an action and receive the next observation.

        Args:
            action: CodeDebugAction with patched code and optional metadata

        Returns:
            Tuple of (observation, reward, done, info)
        """
        resp = self._client.post(
            "/step",
            json={"action": action.model_dump()},
        )
        resp.raise_for_status()
        data = resp.json()

        observation = CodeDebugObservation(**data["observation"])
        reward = data["reward"]
        done = data["done"]
        info = data.get("info", {})

        return observation, reward, done, info

    def get_state(self) -> dict[str, Any]:
        """Get current environment state."""
        resp = self._client.get("/state")
        resp.raise_for_status()
        data = resp.json()
        return data["state"]

    def list_tasks(self) -> list[dict[str, str]]:
        """List all available debugging tasks."""
        resp = self._client.get("/tasks")
        resp.raise_for_status()
        data = resp.json()
        return data["tasks"]

    def get_metrics(self) -> dict[str, Any]:
        """Get aggregate telemetry metrics."""
        resp = self._client.get("/metrics")
        resp.raise_for_status()
        return resp.json()


class CodeDebugLocalClient:
    """
    Direct in-process client — no HTTP overhead.

    Useful for quick prototyping, testing, and training loops
    that don't need the server overhead.
    """

    def __init__(self, **env_kwargs) -> None:
        from codedebug_env.server.environment import CodeDebugEnvironment
        self._env = CodeDebugEnvironment(**env_kwargs)

    def reset(
        self,
        task_id: str | None = None,
        difficulty: str | None = None,
        seed: int | None = None,
    ) -> CodeDebugObservation:
        return self._env.reset(task_id=task_id, difficulty=difficulty, seed=seed)

    def step(
        self, action: CodeDebugAction
    ) -> tuple[CodeDebugObservation, float, bool, dict[str, Any]]:
        return self._env.step(action)

    def get_state(self) -> dict[str, Any]:
        return self._env.get_state()
