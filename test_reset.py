import requests
import time
import subprocess
import os
import sys

proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "codedebug_env.server.app:app", "--port", "7860"])
time.sleep(3)

try:
    resp = requests.post("http://127.0.0.1:7860/reset")
    print("Empty body POST /reset STATUS:", resp.status_code)
    print("Response:", resp.text)
    
    resp2 = requests.post("http://127.0.0.1:7860/reset", json={})
    print("Empty JSON POST /reset STATUS:", resp2.status_code)
    print("Response:", resp2.text)
finally:
    proc.terminate()
