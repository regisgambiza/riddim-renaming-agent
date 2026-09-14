import sqlite3
c = sqlite3.connect('riddim_agent_memory.db')

# Check schema
schema = c.execute("PRAGMA table_info(agent_memory)").fetchall()
print("Schema:", schema)

# Check agent_memory for planner entries
memory = c.execute("SELECT * FROM agent_memory").fetchall()
print(f"Memory entries: {len(memory)}")
for m in memory:
    key = str(m[1]) if len(m) > 1 else ""
    if 'batch' in key.lower() or 'plan' in key.lower() or '064' in key:
        print(f"  Key: {key}")
        data = m[2] if len(m) > 2 else ""
        data_str = str(data)[:200] if data else ""
        print(f"    Data preview: {data_str}")

c.close()