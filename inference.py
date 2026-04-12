import json
import os
from pathlib import Path

from openai import OpenAI

from codedebug_env.client import CodeDebugLocalClient
from codedebug_env.models import CodeDebugAction

def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())

load_local_env()

API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")

BENCHMARK = "codedebug-rl"
MAX_STEPS = 10
MAX_TOTAL_REWARD = 10.0  # Used for mapping cumulative rewards to 0.0-1.0
SUCCESS_SCORE_THRESHOLD = 0.99

TASKS = [
    "builtin_001_fizzbuzz",
    "builtin_002_binary_search",
    "builtin_003_flatten_nested",
]

SYSTEM_PROMPT = """You are an expert Python debugger.
You receive buggy Python code and return the patched full file inside a single ```python block.
Make the smallest reliable change to pass tests."""

if not HF_TOKEN:
    raise ValueError("HF_TOKEN or OPENAI_API_KEY must be set")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    error_value = "null" if error is None else str(error).replace("\n", " ").replace("\r", " ")
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    reward_values = ",".join(f"{r:.2f}" for r in rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={reward_values}", flush=True)

def extract_code_block(reply: str) -> str:
    if "```python" in reply:
        return reply.split("```python", 1)[1].split("```", 1)[0].strip()
    if "```" in reply:
        return reply.split("```", 1)[1].split("```", 1)[0].strip()
    return reply.strip()

def run_task(task_id: str) -> None:
    env = CodeDebugLocalClient(max_steps=MAX_STEPS)
    
    rewards: list[float] = []
    history: list[str] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(task_id=task_id)
        
        for step in range(1, MAX_STEPS + 1):
            if observation.done:
                break
                
            steps_taken = step
            reward = 0.0
            done = False
            error = None
            action_name = "submit_patch_full_replace"
            
            prompt = f"Task: {observation.instruction}\nCurrent code:\n```python\n{observation.current_code}\n```\nTest output:\n{observation.test_output}\n"

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                )
                reply = completion.choices[0].message.content or ""
                patched_code = extract_code_block(reply)
                
                action = CodeDebugAction(patched_code=patched_code, patch_format="full_replace")
                observation, step_reward, done, info = env.step(action)
                reward = step_reward
                
                # Fast track solved states success 
                if done and observation.done_reason == "solved":
                    reward = max(1.0, reward)

            except Exception as exc:
                done = True
                error = str(exc)
                print(f"[DEBUG] Model request failed: {exc}", flush=True)

            rewards.append(reward)
            log_step(step=step, action=action_name, reward=reward, done=done, error=error)
            history.append(f"Step {step}: {action_name} -> reward {reward:+.2f}")

            if done:
                break

        # Compute strictly clamped score [0.0, 1.0] as mandated by the rules
        raw_score = sum(rewards) / len(rewards) if rewards else 0.0
        score = min(max(raw_score, 0.0), 1.0)
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

def main() -> None:
    for task_id in TASKS:
        run_task(task_id)

if __name__ == "__main__":
    main()
