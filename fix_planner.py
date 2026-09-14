with open('riddim_agent_renamimg.py', 'r') as f:
    content = f.read()

old = '            content = message.get("content", "").strip()\n\n            if "```" in content:'

new = '            content = message.get("content", "").strip()\n            ai_log(f"PLANNER: raw response (first 200 chars): {content[:200]}")\n\n            if "```" in content:'

if old in content:
    content = content.replace(old, new, 1)
    with open('riddim_agent_renamimg.py', 'w') as f:
        f.write(content)
    print('OK')
else:
    print('NOT FOUND')