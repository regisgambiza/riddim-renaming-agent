import sqlite3

conn = sqlite3.connect('riddim_agent_memory.db')
cursor = conn.cursor()
cursor.execute('SELECT fingerprint, path, status, last_decision, confidence FROM tracks WHERE status != "completed"')
rows = cursor.fetchall()
print('Non-completed tracks:')
for r in rows:
    print(r)
conn.close()