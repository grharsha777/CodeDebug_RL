import os
import json
import traceback
from openai import OpenAI
from codedebug_env.client import CodeDebugLocalClient
from codedebug_env.models import CodeDebugAction

# Read environment variables with defaults where required
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# Initialize OpenAI client
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

SYSTEM_PROMPT = """You are an expert Python software developer and debugger.
You will be provided with a buggy Python file, its test suite, and the test execution output.
Your goal is to repair the code so that all tests pass.

Return ONLY the full exact replacement code wrapped in a python code block like this:
```python
def my_func():
    pass
```
Do not provide a unified diff. Provide the entire fixed source code.
Make minimal changes to fix the logic. Do not change the function signature."""

def run_task(task_id: str, env_name: str = "codedebug-rl"):
    env = CodeDebugLocalClient(max_steps=10)
    
    # Reset
    obs = env.reset(task_id=task_id)
    print(f"[START] task={task_id} env={env_name} model={MODEL_NAME}")
    
    done = False
    rewards = []
    success = False
    step_num = 0
    
    try:
        while not done and step_num < obs.max_steps:
            step_num += 1
            error_msg = "null"
            reward = 0.0
            
            try:
                # Construct prompt
                prompt = f"""
Task: {obs.instruction}

Current Code:
```python
{obs.current_code}
```

Test Feedback:
{obs.test_output}

Failed Tests: {json.dumps(obs.failed_tests, indent=2)}
"""
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                )
                raw_reply = response.choices[0].message.content or ""
                
                # Extract code block if present
                patched_code = raw_reply
                if "```python" in raw_reply:
                    patched_code = raw_reply.split("```python")[1].split("```")[0].strip()
                elif "```" in raw_reply:
                    patched_code = raw_reply.split("```")[1].split("```")[0].strip()
                
                action = CodeDebugAction(patched_code=patched_code)
                obs, reward, done, info = env.step(action)
                
                success = (obs.done_reason == "solved")
                rewards.append(f"{reward:.2f}")
                
                action_str = "submit_patch_full_replace"

            except Exception as e:
                error_msg = '"' + str(e).replace('"', "'").replace("\n", " ") + '"'
                done = True
                action_str = "error_action"
            
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={str(done).lower()} error={error_msg}")
            
    except Exception as e:
         pass # handled in loop
         
    reward_str = ",".join(rewards) if rewards else "0.00"
    print(f"[END] success={str(success).lower()} steps={step_num} rewards={reward_str}")

def main():
    tasks = ["builtin_001_fizzbuzz", "builtin_002_binary_search", "builtin_003_flatten_nested"]
    for task in tasks:
        run_task(task)

if __name__ == "__main__":
    main()
