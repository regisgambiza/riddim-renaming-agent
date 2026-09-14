import subprocess, time, sys
from pathlib import Path

print("Starting llama-server...")
proc = subprocess.Popen(
    ['C:\\Tools\\llama.cpp\\llama-server.exe',
     '-m', r'C:\Users\regis\.lmstudio\models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf',
     '--host', '127.0.0.1',
     '--port', '8099',
     '-c', '32768',
     '-ngl', '35'],
    stdout=sys.stderr,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
)

print(f"PID: {proc.pid}")
print("Waiting for server to be ready...")

import requests
deadline = time.time() + 300
while time.time() < deadline:
    if proc.poll() is not None:
        print(f"Server exited with code {proc.returncode}")
        break
    try:
        r = requests.get('http://127.0.0.1:8099/health', timeout=2)
        if r.status_code == 200:
            print("Server is ready!")
            break
    except:
        pass
    time.sleep(1)

if proc.poll() is None:
    print("Server is running. Testing chat completion...")
    payload = {
        'model': 'local-model',
        'messages': [{'role': 'user', 'content': 'Return JSON: {"test": "ok"}'}],
        'temperature': 0.1,
        'max_tokens': 100
    }
    r = requests.post('http://127.0.0.1:8099/v1/chat/completions', json=payload, timeout=120)
    print(f"Status: {r.status_code}")
    print(r.text[:500])
    proc.terminate()
    proc.wait()