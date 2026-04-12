import httpx

BASE = "https://grharsha777-codedebug-rl.hf.space"

print("Testing live HF Space endpoints...\n")

# 1. Health check
r = httpx.get(f"{BASE}/health", timeout=15)
print(f"GET /health -> {r.status_code}: {r.json()}")

# 2. POST /reset with no body (exactly what the validator does)
r = httpx.post(f"{BASE}/reset", timeout=60)
print(f"POST /reset (no body) -> {r.status_code}")
if r.status_code == 200:
    data = r.json()
    obs = data.get("observation", {})
    print(f"  task_id: {obs.get('task_id')}")
    print(f"  done: {obs.get('done')}")
    print(f"  step_index: {obs.get('step_index')}")
    print("  PASS: POST /reset returned 200")
else:
    print(f"  FAIL: {r.text[:500]}")

# 3. POST /reset with empty JSON body
r2 = httpx.post(f"{BASE}/reset", json={}, timeout=60)
print(f"POST /reset (empty JSON) -> {r2.status_code}")
if r2.status_code == 200:
    print("  PASS: POST /reset with empty JSON returned 200")
else:
    print(f"  FAIL: {r2.text[:200]}")

# 4. GET /state
r3 = httpx.get(f"{BASE}/state", timeout=15)
print(f"GET /state -> {r3.status_code}")
if r3.status_code == 200:
    print("  PASS: GET /state returned 200")

# 5. GET /tasks
r4 = httpx.get(f"{BASE}/tasks", timeout=15)
print(f"GET /tasks -> {r4.status_code}: {r4.json().get('count')} tasks")
