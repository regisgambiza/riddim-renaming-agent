# Riddim Jellyfin AI Agent

Autonomous music-library cleanup agent for Jellyfin. Renames tracks to `NN - Artist - Title.ext` format and fixes embedded metadata using a local LLM.

## Features

- Batch-level planning + per-track analysis
- Forces Jellyfin dash format (NN - Artist - Title)
- Auto-detects dot-format filenames (`NN. Artist - Title`) for correction
- Auto-assigns missing track numbers
- Persistent memory with decision tracking
- Local LLM via llama.cpp (Qwen3.6)

## Requirements

- Python 3.10+
- llama.cpp server running on port 8099
- Qwen3.6-35B-A3B GGUF model
- mutagen for metadata handling

## Usage

```powershell
python riddim_agent_renamimg.py
```

## Configuration

Edit constants at the top of `riddim_agent_renamimg.py`:
- `ROOT_FOLDER` — music collection root
- `MODEL_PATH` — GGUF model path
- `LLAMA_SERVER_EXE` — llama-server binary
- `SERVER_PORT` — llama.cpp port
- `AUTO_APPLY` — apply changes without confirmation

## Project Structure

- `riddim_agent_renamimg.py` — main agent script
- `riddim_agent_memory.db` — SQLite decision/fingerprint storage (runtime)
- `check_db.py` — inspect memory database
- `check_memory.py` — check memory state
- `test_llm.py` / `test_server.py` — LLM and server tests
