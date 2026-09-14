import requests, json, time

# Test the health endpoint
r = requests.get('http://127.0.0.1:8099/health', timeout=5)
print(f'Health: {r.status_code}')

# Test models
r = requests.get('http://127.0.0.1:8099/v1/models', timeout=5)
print(f'Models: {r.status_code}')

# Test a simple chat completion
payload = {
    'model': 'local-model',
    'messages': [{'role': 'user', 'content': 'Return JSON: {"test": "ok"}'}],
    'temperature': 0.1,
    'max_tokens': 100
}
print('Testing chat completion...')
start = time.time()
r = requests.post('http://127.0.0.1:8099/v1/chat/completions', json=payload, timeout=60)
print(f'Chat: {r.status_code} in {time.time()-start:.1f}s')
print(r.text[:1000])