"""
Tests for the client-server contract — verifies API models round-trip correctly.
"""

import pytest
from fastapi.testclient import TestClient

from codedebug_env.server.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["environment"] == "codedebug-rl"


class TestResetEndpoint:
    def test_reset_default(self, client):
        resp = client.post("/reset", json={})
        assert resp.status_code == 200
        data = resp.json()
        obs = data["observation"]
        assert "task_id" in obs
        assert "current_code" in obs
        assert obs["done"] is False

    def test_reset_with_task_id(self, client):
        resp = client.post("/reset", json={"task_id": "builtin_001_fizzbuzz"})
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs["task_id"] == "builtin_001_fizzbuzz"

    def test_reset_with_difficulty(self, client):
        resp = client.post("/reset", json={"difficulty": "easy"})
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert obs["metadata"]["difficulty"] == "easy"

    def test_reset_invalid_task_returns_400(self, client):
        resp = client.post("/reset", json={"task_id": "nonexistent"})
        assert resp.status_code == 400


class TestStepEndpoint:
    def test_step_without_reset_returns_400(self, client):
        # Need to reset first in a fresh app instance
        # Since TestClient shares state, first reset then test step
        pass

    def test_full_episode_workflow(self, client):
        """Test complete reset → step → done workflow."""
        # Reset
        resp = client.post("/reset", json={"task_id": "builtin_001_fizzbuzz"})
        assert resp.status_code == 200

        # Step with correct fix
        correct_code = (
            'def fizzbuzz(n: int) -> list[str]:\n'
            '    """Return FizzBuzz sequence from 1 to n."""\n'
            '    result = []\n'
            '    for i in range(1, n + 1):\n'
            '        if i % 3 == 0 and i % 5 == 0:\n'
            '            result.append("FizzBuzz")\n'
            '        elif i % 3 == 0:\n'
            '            result.append("Fizz")\n'
            '        elif i % 5 == 0:\n'
            '            result.append("Buzz")\n'
            '        else:\n'
            '            result.append(str(i))\n'
            '    return result\n'
        )
        resp = client.post(
            "/step",
            json={
                "action": {
                    "patched_code": correct_code,
                    "reasoning": "Fixed type error",
                    "patch_format": "full_replace",
                }
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data
        assert "reward" in data
        assert "done" in data
        assert data["reward"] > 0
        assert data["done"] is True
        assert data["observation"]["done_reason"] == "solved"

    def test_step_returns_reward_breakdown(self, client):
        """Verify reward breakdown is included in step response."""
        client.post("/reset", json={"task_id": "builtin_001_fizzbuzz"})
        resp = client.post(
            "/step",
            json={"action": {"patched_code": "def fizzbuzz(n): return []"}},
        )
        assert resp.status_code == 200
        obs = resp.json()["observation"]
        assert "reward_breakdown" in obs
        assert "total" in obs["reward_breakdown"]


class TestStateEndpoint:
    def test_state_before_reset(self, client):
        # State depends on app-level env which may already be reset
        resp = client.get("/state")
        assert resp.status_code == 200
        assert "state" in resp.json()

    def test_state_after_reset(self, client):
        client.post("/reset", json={"task_id": "builtin_001_fizzbuzz"})
        resp = client.get("/state")
        assert resp.status_code == 200
        state = resp.json()["state"]
        assert state["task_id"] == "builtin_001_fizzbuzz"


class TestTasksEndpoint:
    def test_list_tasks(self, client):
        resp = client.get("/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] >= 3
        assert len(data["tasks"]) >= 3


class TestMetricsEndpoint:
    def test_metrics_available(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "aggregate" in data
