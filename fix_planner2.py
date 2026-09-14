with open('riddim_agent_renamimg.py', 'r') as f:
    content = f.read()

# Add more logging in the error path
old_error = 'error_log(\n                        f"Planner returned invalid JSON for {track[\'filename\']}: {exc}\\n"\n                        f"Content (first 200 chars): {content[:200]}"\n                    )'
new_error = 'ai_log(f"PLANNER: JSON parse error for {track[\'filename\']}: {exc}")\n                    error_log(\n                        f"Planner returned invalid JSON for {track[\'filename\']}: {exc}\\n"\n                        f"Content (first 500 chars): {content[:500]}"\n                    )'

if old_error in content:
    content = content.replace(old_error, new_error, 1)
    print('Error logging updated')
else:
    print('Error logging NOT FOUND')

# Also add logging for the "No JSON object found" case
old_no_json = 'error_log(\n                        f"Planner returned invalid JSON for {track[\'filename\']}: No JSON object found\\n"\n                        f"Content (first 200 chars): {content[:200]}"\n                    )'
new_no_json = 'ai_log(f"PLANNER: No JSON object found for {track[\'filename\']}")\n                    error_log(\n                        f"Planner returned invalid JSON for {track[\'filename\']}: No JSON object found\\n"\n                        f"Content (first 500 chars): {content[:500]}"\n                    )'

if old_no_json in content:
    content = content.replace(old_no_json, new_no_json, 1)
    print('No-JSON logging updated')
else:
    print('No-JSON logging NOT FOUND')

with open('riddim_agent_renamimg.py', 'w') as f:
    f.write(content)
print('Done')