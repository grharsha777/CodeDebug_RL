from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

from codedebug_env.client import CodeDebugLocalClient
from codedebug_env.models import CodeDebugAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
BENCHMARK = "codedebug-rl"
RESULTS_PATH = Path(__file__).resolve().parent / "baseline_results.json"
TASKS = [
    "builtin_001_fizzbuzz",
    "builtin_002_binary_search",
    "builtin_003_flatten_nested",
]

SYSTEM_PROMPT = """You are an expert Python debugger.
You receive buggy Python code, structured task instructions, and test feedback.
Return only the full fixed Python source inside a single ```python fenced block.
Make the smallest reliable change that improves the tests."""

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN or OPENAI_API_KEY must be set")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def log_start(task: str) -> str:
    line = f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}"
    print(line, flush=True)
    return line


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> str:
    error_value = "null" if error is None else error.replace("\n", " ").replace("\r", " ")
    line = f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}"
    print(line, flush=True)
    return line


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> str:
    reward_values = ",".join(f"{reward:.2f}" for reward in rewards) if rewards else "0.00"
    line = f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={reward_values}"
    print(line, flush=True)
    return line


def build_prompt(observation) -> str:
    failed_tests = json.dumps(observation.failed_tests, indent=2)
    return f"""Task: {observation.instruction}

Current code:
```python
{observation.current_code}
```

Execution status: {observation.execution_status}
Test summary: {json.dumps(observation.test_summary, indent=2)}
Failed tests:
{failed_tests}

Pytest output:
{observation.test_output}
"""


def extract_code_block(reply: str) -> str:
    if "```python" in reply:
        return reply.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in reply:
        return reply.split("```", 1)[1].split("```", 1)[0].strip()
    return reply.strip()


def compute_score(observation, info: dict[str, object]) -> float:
    if "grader_score" in info:
        return round(float(info["grader_score"]), 4)
    summary = observation.test_summary or {}
    total = int(summary.get("total", 0))
    passed = int(summary.get("passed", 0))
    return round((passed / total) if total else 0.0, 4)


def run_task(task_id: str) -> dict[str, object]:
    env = CodeDebugLocalClient(max_steps=10)
    observation = env.reset(task_id=task_id)
    rewards: list[float] = []
    score = compute_score(observation, {})
    success = False
    steps = 0
    transcript: list[str] = []

    transcript.append(log_start(task_id))

    try:
        while steps < observation.max_steps and not observation.done:
            steps += 1
            reward = 0.0
            done = False
            error: str | None = None
            action_name = "submit_patch_full_replace"

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    temperature=0.0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_prompt(observation)},
                    ],
                )
                reply = response.choices[0].message.content or ""
                patched_code = extract_code_block(reply)
                action = CodeDebugAction(
                    patched_code=patched_code,
                    patch_format="full_replace",
                )
                observation, reward, done, info = env.step(action)
                score = compute_score(observation, info)
                success = bool(done and observation.done_reason == "solved")
            except Exception as exc:
                done = True
                error = str(exc)

            rewards.append(reward)
            transcript.append(log_step(steps, action_name, reward, done, error))

            if done:
                break
    finally:
        env.close()
        transcript.append(log_end(success, steps, score, rewards))

    return {
        "task_id": task_id,
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": [round(reward, 4) for reward in rewards],
        "transcript": transcript,
    }


def main() -> None:
    results = [run_task(task_id) for task_id in TASKS]
    average_score = round(sum(item["score"] for item in results) / len(results), 4) if results else 0.0
    RESULTS_PATH.write_text(
        json.dumps(
            {
                "benchmark": BENCHMARK,
                "model": MODEL_NAME,
                "api_base_url": API_BASE_URL,
                "results": results,
                "average_score": average_score,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
