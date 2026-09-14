import os
import sys
import json
import re
import time
import sqlite3
import hashlib
import signal
import subprocess
from pathlib import Path
from typing import Any

import requests
from mutagen import File as MutagenFile


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"C:\Users\regis\.lmstudio\models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf"

# Change this if llama-server.exe is somewhere else.
LLAMA_SERVER_EXE = r"C:\Tools\llama.cpp\llama-server.exe"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8099

# Your riddim collection
ROOT_FOLDER = Path(r"E:\Sample_Riddim_Collection")

# Persistent agent memory
STATE_DB = Path("riddim_agent_memory.db")

# Default = no changes until approved
AUTO_APPLY = True

# Number of agent iterations before forcing a fresh context
MAX_AGENT_STEPS = 40

# AI parameters
TEMPERATURE = 0.1
MAX_TOKENS = 4000

# llama.cpp settings
CONTEXT_SIZE = 32768
N_GPU_LAYERS = "auto"


# ============================================================
# CONSOLE LOGGING
# ============================================================

def log(message: str, prefix: str = "AGENT"):
    safe = message.encode("ascii", "replace").decode("ascii")
    print(f"[{time.strftime('%H:%M:%S')}] {prefix} > {safe}", flush=True)


def ai_log(message: str):
    log(message, "AI")


def tool_log(name: str, arguments: dict):
    compact = json.dumps(arguments, ensure_ascii=False)
    log(f"{name}({compact})", "TOOL")


def result_log(message: str):
    log(message, "RESULT")


def error_log(message: str):
    log(message, "ERROR")


# ============================================================
# PERSISTENT MEMORY
# ============================================================

class Memory:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    def create_tables(self):
        self.conn.executescript("""
        CREATE TABLE IF NOT EXISTS tracks (
            id INTEGER PRIMARY KEY,
            fingerprint TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL,
            riddim TEXT,
            year TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            last_decision TEXT,
            confidence TEXT,
            artist TEXT,
            title TEXT,
            album TEXT,
            original_path TEXT,
            new_path TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS operations (
            id INTEGER PRIMARY KEY,
            batch_id TEXT NOT NULL,
            fingerprint TEXT NOT NULL,
            operation TEXT NOT NULL,
            original_path TEXT,
            new_path TEXT,
            original_metadata TEXT,
            new_metadata TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS agent_memory (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY,
            fingerprint TEXT UNIQUE NOT NULL,
            path TEXT NOT NULL,
            reason TEXT NOT NULL,
            proposal TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            resolved INTEGER DEFAULT 0
        );
        """)
        self.conn.commit()

    def get_track(self, fingerprint):
        return self.conn.execute(
            "SELECT * FROM tracks WHERE fingerprint=?",
            (fingerprint,)
        ).fetchone()

    def upsert_track(self, data):
        self.conn.execute("""
        INSERT INTO tracks (
            fingerprint, path, riddim, year, status,
            last_decision, confidence, artist, title,
            album, original_path, new_path
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fingerprint) DO UPDATE SET
            path=excluded.path,
            riddim=excluded.riddim,
            year=excluded.year,
            status=excluded.status,
            last_decision=excluded.last_decision,
            confidence=excluded.confidence,
            artist=excluded.artist,
            title=excluded.title,
            album=excluded.album,
            original_path=excluded.original_path,
            new_path=excluded.new_path,
            updated_at=CURRENT_TIMESTAMP
        """, (
            data["fingerprint"],
            data["path"],
            data.get("riddim"),
            data.get("year"),
            data.get("status", "pending"),
            data.get("last_decision"),
            data.get("confidence"),
            data.get("artist"),
            data.get("title"),
            data.get("album"),
            data.get("original_path"),
            data.get("new_path"),
        ))
        self.conn.commit()

    def record_operation(
        self,
        batch_id,
        fingerprint,
        operation,
        original_path,
        new_path,
        original_metadata,
        new_metadata
    ):
        self.conn.execute("""
        INSERT INTO operations (
            batch_id,
            fingerprint,
            operation,
            original_path,
            new_path,
            original_metadata,
            new_metadata
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id,
            fingerprint,
            operation,
            original_path,
            new_path,
            json.dumps(original_metadata, ensure_ascii=False),
            json.dumps(new_metadata, ensure_ascii=False),
        ))
        self.conn.commit()

    def add_review(self, fingerprint, path, reason, proposal):
        self.conn.execute("""
        INSERT INTO reviews (
            fingerprint, path, reason, proposal
        )
        VALUES (?, ?, ?, ?)
        ON CONFLICT(fingerprint) DO UPDATE SET
            reason=excluded.reason,
            proposal=excluded.proposal,
            resolved=0
        """, (
            fingerprint,
            path,
            reason,
            json.dumps(proposal, ensure_ascii=False)
        ))
        self.conn.commit()

    def set_memory(self, key, value):
        self.conn.execute("""
        INSERT INTO agent_memory(key, value)
        VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET
            value=excluded.value,
            updated_at=CURRENT_TIMESTAMP
        """, (key, value))
        self.conn.commit()

    def get_memory(self, key):
        row = self.conn.execute(
            "SELECT value FROM agent_memory WHERE key=?",
            (key,)
        ).fetchone()

        return row["value"] if row else None

    def close(self):
        self.conn.close()


# ============================================================
# FILE FINGERPRINT
# ============================================================

def fingerprint(path: Path) -> str:
    """
    Stable-ish identity based on path + size + modification time.
    """
    stat = path.stat()

    raw = (
        str(path.resolve())
        + "|"
        + str(stat.st_size)
        + "|"
        + str(stat.st_mtime_ns)
    )

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ============================================================
# METADATA
# ============================================================

def read_metadata(path: Path) -> dict:
    audio = MutagenFile(str(path), easy=True)

    if audio is None:
        return {
            "error": "Unable to read audio metadata"
        }

    def first(key):
        value = audio.get(key)
        if isinstance(value, list) and value:
            return value[0]
        return value

    return {
        "title": first("title"),
        "artist": first("artist"),
        "album": first("album"),
        "albumartist": first("albumartist"),
        "genre": first("genre"),
        "date": first("date"),
        "tracknumber": first("tracknumber"),
        "discnumber": first("discnumber"),
    }


def write_metadata(path: Path, metadata: dict):
    audio = MutagenFile(str(path), easy=True)

    if audio is None:
        raise RuntimeError(f"Cannot open {path}")

    mapping = {
        "title": metadata.get("title"),
        "artist": metadata.get("artist"),
        "album": metadata.get("album"),
        "albumartist": metadata.get("albumartist"),
        "genre": metadata.get("genre"),
        "date": metadata.get("date"),
        "tracknumber": metadata.get("tracknumber"),
        "discnumber": metadata.get("discnumber"),
    }

    for key, value in mapping.items():
        if value is not None:
            audio[key] = [str(value)]

    audio.save()


# ============================================================
# RIDDIM CONTEXT
# ============================================================

def get_riddim_context(path: Path) -> dict:
    """
    Assumes:
        ROOT/year/riddim/track.mp3

    But doesn't require this exact structure.
    """

    riddim = path.parent.name

    year = None

    for parent in path.parents:
        if parent == ROOT_FOLDER.parent:
            break

        if parent.name.isdigit() and len(parent.name) == 4:
            year = parent.name
            break

    return {
        "riddim": riddim,
        "year": year,
    }


def list_riddim_tracks(riddim_folder: Path) -> list[dict]:
    tracks = []

    for p in sorted(riddim_folder.iterdir()):
        if not p.is_file():
            continue

        if p.suffix.lower() not in {
            ".mp3", ".flac", ".m4a", ".aac",
            ".ogg", ".opus", ".wav"
        }:
            continue

        metadata = read_metadata(p)

        tracks.append({
            "filename": p.name,
            "metadata": metadata
        })

    return tracks


# ============================================================
# RENAME
# ============================================================

def safe_filename(text: str) -> str:
    forbidden = '<>:"/\\|?*'

    for char in forbidden:
        text = text.replace(char, "")

    text = " ".join(text.split())

    return text.strip(" .")


def make_filename(track_number, artist, title, extension):
    artist = safe_filename(artist)
    title = safe_filename(title)

    return f"{track_number:02d} - {artist} - {title}{extension}"


# ============================================================
# LOCAL SEARCH TOOL
# ============================================================

def search_local_tracks(query: str, limit: int = 20):
    """
    Searches already indexed filenames/metadata from the filesystem.

    This gives the agent another tool it can use when deciding
    whether a suspicious track matches another known track.
    """

    query_lower = query.lower()
    results = []

    for path in ROOT_FOLDER.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".mp3", ".flac", ".m4a", ".aac",
            ".ogg", ".opus", ".wav"
        }:
            continue

        if query_lower in path.name.lower():
            results.append(str(path))

            if len(results) >= limit:
                break

    return results


# ============================================================
# AGENT TOOLS
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_metadata",
            "description": "Read embedded audio metadata from a track.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_riddim_tracks",
            "description": "List all audio tracks and metadata in the current riddim folder.",
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string"
                    }
                },
                "required": ["folder"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_local_tracks",
            "description": "Search the existing music collection for filenames matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string"
                    },
                    "limit": {
                        "type": "integer"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "propose_track_change",
            "description": "Create a proposal to rename and retag a track. Does not modify the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string"
                    },
                    "artist": {
                        "type": "string"
                    },
                    "title": {
                        "type": "string"
                    },
                    "track_number": {
                        "type": "integer"
                    },
                    "confidence": {
                        "type": "string",
                        "enum": [
                            "HIGH",
                            "MEDIUM",
                            "LOW"
                        ]
                    },
                    "reason": {
                        "type": "string"
                    }
                },
                "required": [
                    "path",
                    "artist",
                    "title",
                    "track_number",
                    "confidence",
                    "reason"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "plan_riddim_batch",
            "description": (
                "Create a batch-level cleanup plan for an entire riddim. "
                "Analyze all tracks together, identify naming patterns, "
                "metadata quality, ambiguous tracks, likely artist/title "
                "parsing rules, duplicates, and tracks requiring research. "
                "Do not modify files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string"
                    },
                    "riddim": {
                        "type": "string"
                    },
                    "year": {
                        "type": "string"
                    },
                    "tracks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "filename": {"type": "string"},
                                "metadata": {"type": "object"}
                            },
                            "required": [
                                "path",
                                "filename",
                                "metadata"
                            ]
                        }
                    }
                },
                "required": [
                    "folder",
                    "riddim",
                    "year",
                    "tracks"
                ]
            }
        }
    }
]


# ============================================================
# TOOL EXECUTION
# ============================================================

class ToolExecutor:

    def __init__(self, memory: Memory):
        self.memory = memory
        self.pending_proposals = []

    def execute(self, name: str, args: dict, context: dict):

        tool_log(name, args)

        if name == "read_metadata":

            path = Path(args["path"])

            result = read_metadata(path)

            result_log(json.dumps(result, ensure_ascii=False))

            return result

        if name == "list_riddim_tracks":

            folder = Path(args["folder"])

            result = list_riddim_tracks(folder)

            result_log(f"{len(result)} tracks found")

            return result

        if name == "search_local_tracks":

            result = search_local_tracks(
                args["query"],
                args.get("limit", 20)
            )

            result_log(f"{len(result)} matches")

            return result

        if name == "propose_track_change":

            path = Path(args["path"])

            proposal = {
                "path": str(path),
                "artist": args["artist"],
                "title": args["title"],
                "track_number": args["track_number"],
                "confidence": args["confidence"],
                "reason": args["reason"],
            }

            self.pending_proposals.append(proposal)

            ai_log(
                f"Proposal created: "
                f"{args['artist']} - {args['title']} "
                f"[{args['confidence']}]"
            )

            return {
                "status": "proposal_created",
                "proposal": proposal
            }

        raise ValueError(f"Unknown tool: {name}")


# ============================================================
# LLAMA.CPP SERVER
# ============================================================

class LlamaServer:

    def __init__(self):
        self.process = None

        self.base_url = (
            f"http://{SERVER_HOST}:{SERVER_PORT}"
        )

    def start(self):

        if not Path(LLAMA_SERVER_EXE).exists():
            raise FileNotFoundError(
                f"llama-server.exe not found:\n"
                f"{LLAMA_SERVER_EXE}\n\n"
                f"Change LLAMA_SERVER_EXE in the script."
            )

        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(
                f"Model file not found:\n"
                f"{MODEL_PATH}\n\n"
                f"Change MODEL_PATH in the script."
            )

        log("Starting llama.cpp server...")

        cmd = [
            LLAMA_SERVER_EXE,
            "-m", MODEL_PATH,
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "-c", str(CONTEXT_SIZE),
            "-ngl", str(N_GPU_LAYERS),
            "--reasoning", "off",
        ]

        log(" ".join(f'"{x}"' if " " in x else x for x in cmd))

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )

        self.wait_until_ready()

    def wait_until_ready(self):

        log("Waiting for llama.cpp...")

        startup_timeout = 600
        deadline = time.time() + startup_timeout

        while time.time() < deadline:

            if self.process.poll() is not None:
                error_log(
                    "llama.cpp server stopped unexpectedly."
                )
                raise RuntimeError(
                    "llama.cpp server stopped unexpectedly."
                )

            try:
                r = requests.get(
                    f"{self.base_url}/health",
                    timeout=2
                )

                if r.status_code == 200:
                    log("llama.cpp server is ready.")
                    return

            except requests.RequestException:
                pass

            time.sleep(1)

        raise TimeoutError(
            f"llama.cpp server did not become ready "
            f"within {startup_timeout}s."
        )

    def stop(self):

        if self.process is None:
            return

        if self.process.poll() is not None:
            return

        log("Stopping llama.cpp server...")

        try:
            self.process.terminate()
            self.process.wait(timeout=10)

        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)

        log("llama.cpp server stopped.")


# ============================================================
# LLM CLIENT
# ============================================================

class LLM:

    def __init__(self, server: LlamaServer):
        self.server = server

    def chat(self, messages, tools=None, response_format=None):

        payload = {
            "model": "local-model",
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }

        if response_format is not None:
            payload["response_format"] = response_format

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = requests.post(
            f"{self.server.base_url}/v1/chat/completions",
            json=payload,
            timeout=600
        )

        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]

        # Handle models that put output in reasoning_content instead of content
        content = message.get("content", "")
        if not content and message.get("reasoning_content"):
            content = message["reasoning_content"]
            message["content"] = content

        return message


# ============================================================
# BATCH PLANNER
# ============================================================

def collection_fingerprint(riddim_folder: Path) -> str:
    entries = []

    for p in sorted(riddim_folder.iterdir()):
        if p.is_file():
            stat = p.stat()
            entries.append(
                f"{p.name}|{stat.st_size}|{stat.st_mtime_ns}"
            )

    raw = "\n".join(entries)

    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class RiddimPlanner:

    def __init__(self, llm, memory):
        self.llm = llm
        self.memory = memory

    def _plan_key(self, folder, year):
        return hashlib.sha256(
            f"{folder}|{year}".encode()
        ).hexdigest()

    def _load_plan(self, key):
        value = self.memory.get_memory(
            f"batch_plan:{key}"
        )

        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

    def _save_plan(self, key, plan):
        self.memory.set_memory(
            f"batch_plan:{key}",
            json.dumps(plan, ensure_ascii=False)
        )

    def create_plan(self, folder, riddim, year, tracks):
        ai_log(
            f"PLANNER: analyzing riddim one track at a time: {riddim}"
        )

        current_fp = collection_fingerprint(folder)
        key = self._plan_key(folder, year)
        existing = self._load_plan(key)

        if existing:
            stored_fp = existing.get(
                "collection_fingerprint"
            )

            if stored_fp == current_fp:
                ai_log(
                    "PLANNER: existing plan is valid. "
                    "Resuming."
                )
                return existing

            ai_log(
                "PLANNER: contents changed. "
                "Re-planning."
            )

        compact_tracks = []
        for track in tracks:
            metadata = track.get("metadata") or {}
            compact_metadata = {}

            for key in ("title", "artist", "album", "albumartist", "genre", "date", "tracknumber", "discnumber"):
                value = metadata.get(key)
                if value not in (None, "", [], {}):
                    compact_metadata[key] = value

            compact_tracks.append({
                "path": track.get("path"),
                "filename": track.get("filename"),
                "metadata": compact_metadata,
            })

        plan = {
            "riddim": riddim,
            "year": year,
            "summary": (
                f"Analyzed {len(compact_tracks)} tracks individually "
                f"to reduce local-model load."
            ),
            "strategy": "One-track-at-a-time planning with lightweight prompts.",
            "tracks": [],
            "batch_warnings": [],
        }

        for idx, track in enumerate(compact_tracks, start=1):
            ai_log(
                f"PLANNER: analyzing track {idx}/{len(compact_tracks)}: "
                f"{track['filename']}"
            )

            messages = [
                {
                    "role": "system",
                    "content": """
You are the single-track planner for a Jellyfin music-cleanup agent.

Analyze ONE track at a time. Keep the prompt tiny so the local model can respond reliably.

The folders are already correctly named.
DO NOT rename folders.

ABSOLUTE FORMAT RULES:
- Return exactly one JSON object and nothing else.
- Do not wrap the JSON in markdown code fences.
- Do not include explanations, commentary, or prose.
- Do not include trailing commas or comments.

JELLYFIN FILENAME STANDARD (MUST follow exactly):
- Format: NN - Artist - Title.ext  (note the DASH between number and artist)
- Example: "01 - Capleton - In Her Heart.mp3"
- Files using dots like "01. Capleton - In Her Heart.mp3" are WRONG and must be cleaned to dashes.
- If the filename uses a dot separator (e.g. "NN. Artist - Title"), action MUST be CLEAN.

Schema:
{
    "action": "KEEP|CLEAN|RESEARCH|NEEDS_REVIEW",
    "artist": "...",
    "title": "...",
    "track_number": 1,
    "confidence": "HIGH|MEDIUM|LOW",
    "reason": "..."
}

Rules:
- Never invent artist or title information.
- If evidence is insufficient, use NEEDS_REVIEW.
- LOW confidence must not result in an automatic modification.
- Keep the reason concise but operational.
- Do NOT include "path" or "filename" in the response object. These fields are for input only. The agent already knows the file paths from its own filesystem scan.
- If the filename does NOT match "NN - Artist - Title.ext" (dash), action MUST be CLEAN.
- A CLEAN action means the track needs to be renamed to the Jellyfin dash format and/or have its metadata updated.
"""
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "folder": str(folder),
                        "riddim": riddim,
                        "year": year,
                        "track": track
                    }, ensure_ascii=False, indent=2)
                }
            ]

            message = self.llm.chat(
                messages,
                tools=None,
                response_format={
                    "type": "json_object"
                }
            )

            content = message.get("content", "").strip()
            ai_log(f"PLANNER: raw response (first 200 chars): {content[:200]}")

            if "```" in content:
                start_idx = content.find("```")
                after_first = content[start_idx + 3:]
                end_idx = after_first.find("```")
                if end_idx != -1:
                    content = after_first[:end_idx].strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
                else:
                    lines = content.split("\n")
                    if lines:
                        content = "\n".join(lines[1:]).strip()

            try:
                decision = json.loads(content)
            except json.JSONDecodeError:
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_str = content[start:end+1]
                    try:
                        decision = json.loads(json_str)
                    except json.JSONDecodeError as exc:
                        error_log(
                            f"Planner returned invalid JSON for {track['filename']}: {exc}\n"
                            f"Content (first 200 chars): {content[:200]}"
                        )
                        decision = {
                            "path": track.get("path"),
                            "filename": track.get("filename"),
                            "action": "NEEDS_REVIEW",
                            "artist": "",
                            "title": "",
                            "track_number": 0,
                            "confidence": "LOW",
                            "reason": "Planner returned invalid JSON; queued for review.",
                        }
                else:
                    ai_log(f"PLANNER: No JSON object found for {track['filename']}")
                    error_log(
                        f"Planner returned invalid JSON for {track['filename']}: No JSON object found\n"
                        f"Content (first 500 chars): {content[:500]}"
                    )
                    decision = {
                        "path": track.get("path"),
                        "filename": track.get("filename"),
                        "action": "NEEDS_REVIEW",
                        "artist": "",
                        "title": "",
                        "track_number": 0,
                        "confidence": "LOW",
                        "reason": "Planner returned invalid JSON; queued for review.",
                    }

            if not isinstance(decision, dict):
                decision = {
                    "path": track.get("path"),
                    "filename": track.get("filename"),
                    "action": "NEEDS_REVIEW",
                    "artist": "",
                    "title": "",
                    "track_number": 0,
                    "confidence": "LOW",
                    "reason": "Planner response was not a track decision.",
                }

            # Defense-in-depth: always use the actual filesystem path.
            # The LLM must never provide file paths; they are owned by the agent.
            decision["path"] = track.get("path")
            decision["filename"] = track.get("filename")
            decision.setdefault("action", "NEEDS_REVIEW")
            decision.setdefault("artist", "")
            decision.setdefault("title", "")
            decision.setdefault("track_number", 0)
            decision.setdefault("confidence", "LOW")
            decision.setdefault("reason", "No supporting evidence found.")
            decision["riddim"] = riddim
            decision["year"] = year

            if not decision.get("filename"):
                decision["filename"] = Path(decision.get("path", "")).name

            plan["tracks"].append(decision)

        plan["collection_fingerprint"] = current_fp
        plan["folder"] = str(folder)
        plan["key"] = key

        self._save_plan(key, plan)

        ai_log(
            f"PLANNER: {len(plan.get('tracks', []))} "
            f"track decisions created"
        )

        return plan


# ============================================================
# LLM CLIENT
# ============================================================

SYSTEM_PROMPT = """
You are an autonomous music-library cleanup agent.

Your job is to clean TRACK FILENAMES and EMBEDDED METADATA
for Jellyfin.

IMPORTANT:
- NEVER rename folders.
- NEVER delete audio files.
- NEVER invent artist or title information.
- If evidence is insufficient, use NEEDS_REVIEW.
- You have tools. Use them when useful.
- Inspect sibling tracks when the current track is ambiguous.
- Existing correct metadata should normally be preserved.
- You are allowed to make decisions, but actual modifications
  require human approval.

Jellyfin target:

Filename:
NN - Artist - Song Title.ext

Metadata:
Title = Song Title
Artist = Artist
Album = riddim folder name
Album Artist = Various Artists
Year = parent year folder
Track Number = NN
Genre = Dancehall

Confidence policy:

HIGH:
    Multiple pieces of evidence agree.

MEDIUM:
    Strong local evidence but incomplete confirmation.

LOW:
    Weak, conflicting, or speculative evidence.

LOW confidence must result in NEEDS_REVIEW.

Do not explain private chain-of-thought.
Instead provide concise operational reasoning:
- what you observed
- what evidence you are using
- why you are calling a tool
- what decision you reached
"""


class RiddimAgent:

    def __init__(self, llm: LLM, memory: Memory):

        self.llm = llm
        self.memory = memory
        self.tools = ToolExecutor(memory)

        # Track number auto-assignment state
        self._track_num_counter = {}      # riddim -> next available number
        self._used_track_nums = {}        # riddim -> set of used numbers

    def process_track(
        self,
        path: Path,
        riddim: str,
        year: str | None
    ):

        fp = fingerprint(path)

        previous = self.memory.get_track(fp)

        if previous:
            if previous["status"] in {"completed", "approved", "compliant"}:
                log(
                    f"Skipping previously processed track: "
                    f"{path.name}"
                )
                return

        metadata = read_metadata(path)

        # Enforce Jellyfin dash format: if filename uses dot separator, force CLEAN.
        if re.match(r'^\d{1,2}\.\s', path.stem):
            m = re.match(r'^(\d{1,2})\.\s*(.+?)\s*-\s*(.+)$', path.stem)
            if m:
                artist = m.group(2).strip()
                title = m.group(3).strip()
                track_num = int(m.group(1))
                self._used_track_nums.setdefault(riddim, set())
                self._used_track_nums[riddim].add(track_num)
                if track_num >= self._track_num_counter.get(riddim, 1):
                    self._track_num_counter[riddim] = track_num + 1
                self.tools.pending_proposals.append({
                    "path": str(path),
                    "artist": artist,
                    "title": title,
                    "track_number": track_num,
                    "confidence": "HIGH",
                    "reason": "Filename uses dot format; Jellyfin standard requires dash format.",
                })
                self.memory.upsert_track({
                    "fingerprint": fp,
                    "path": str(path),
                    "riddim": riddim,
                    "year": year,
                    "status": "pending_review",
                    "last_decision": "CLEAN",
                    "confidence": "HIGH",
                    "artist": artist,
                    "title": title,
                    "album": riddim,
                })
                log(
                    f"Dot-format detected, requiring CLEAN: {path.name}"
                )
                return

        context = {
            "track_path": str(path),
            "filename": path.name,
            "riddim": riddim,
            "year": year,
            "metadata": metadata,
        }

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "task": "Analyze this track",
                        "context": context
                    },
                    ensure_ascii=False,
                    indent=2
                )
            }
        ]

        ai_log(
            f"Investigating: {path.name}"
        )

        message = self.llm.chat(messages, tools=None)

        content = message.get("content", "").strip()
        ai_log(f"Track analysis response: {content[:500]}")

        decision = self._parse_track_response(content)

        action = decision.get("action", "NEEDS_REVIEW")

        if action == "KEEP":
            self.memory.upsert_track({
                "fingerprint": fp,
                "path": str(path),
                "riddim": riddim,
                "year": year,
                "status": "compliant",
                "last_decision": "COMPLIANT",
                "confidence": "HIGH",
                "artist": decision.get("artist"),
                "title": decision.get("title"),
                "album": riddim,
            })
            return

        if action == "NEEDS_REVIEW":
            self.memory.upsert_track({
                "fingerprint": fp,
                "path": str(path),
                "riddim": riddim,
                "year": year,
                "status": "needs_review",
                "last_decision": "NEEDS_REVIEW",
                "confidence": "LOW",
                "reason": decision.get("reason", "Insufficient evidence."),
            })
            return

        if action == "CLEAN":
            artist = decision.get("artist")
            title = decision.get("title")
            track_number = decision.get("track_number")

            # Initialize riddim state if needed
            if riddim not in self._used_track_nums:
                self._used_track_nums[riddim] = set()
                self._track_num_counter[riddim] = 1

            # Handle track number: use provided or auto-assign, check for conflicts
            if track_number:
                if track_number in self._used_track_nums[riddim]:
                    # Conflict! Auto-assign next available number
                    track_number = self._next_track_number(riddim)
                else:
                    self._used_track_nums[riddim].add(track_number)
                    # Update counter if this number is >= current counter
                    if track_number >= self._track_num_counter[riddim]:
                        self._track_num_counter[riddim] = track_number + 1
            else:
                # No track number provided - auto-assign
                track_number = self._next_track_number(riddim)

            if not artist or not title:
                self.memory.upsert_track({
                    "fingerprint": fp,
                    "path": str(path),
                    "riddim": riddim,
                    "year": year,
                    "status": "needs_review",
                    "last_decision": "NEEDS_REVIEW",
                    "confidence": "LOW",
                    "reason": "Clean decision missing required fields.",
                })
                return

            self.tools.pending_proposals.append({
                "path": str(path),
                "artist": artist,
                "title": title,
                "track_number": int(track_number),
                "confidence": decision.get("confidence", "HIGH"),
                "reason": decision.get("reason", "Track needs cleaning."),
            })
            return

        self.memory.upsert_track({
            "fingerprint": fp,
            "path": str(path),
            "riddim": riddim,
            "year": year,
            "status": "needs_review",
            "last_decision": "UNKNOWN_ACTION",
            "confidence": "LOW",
            "reason": f"Unknown action: {action}",
        })

    def _parse_track_response(self, content: str) -> dict:

        # First, try to find structured action patterns
        for pattern in [
            r'"action"\s*:\s*"(\w+)"',
            r"action[:\s]+(\w+)",
            r"Action[:\s]+(\w+)",
        ]:
            m = re.search(pattern, content, re.IGNORECASE)
            if m:
                action = m.group(1).upper()
                if action in ("KEEP", "CLEAN", "NEEDS_REVIEW"):
                    break
        else:
            # No structured action found - analyze prose for compliance indicators
            content_lower = content.lower()
            
            # Check for compliance indicators in prose
            compliance_indicators = [
                "matches.*perfectly",
                "already.*correct",
                "consistent with",
                "no discrepancies",
                "no changes needed",
                "filename.*matches.*metadata",
                "metadata.*matches.*filename",
                "no renaming required",
                "no cleaning required",
                "format.*correct",
                "standard.*followed",
                "jellyfin.*standard",
                "no.*issue",
                "in good order"
            ]
            
            non_compliance_indicators = [
                "needs.*rename",
                "must be renamed",
                "requires.*clean",
                "should be",
                "needs.*clean",
                "dot.*format",
                "dot separator",
                "missing.*track",
                "leading zero",
                "incorrect.*format",
                "not.*consistent",
                "does not match",
                "wrong.*format",
                "rename.*to",
                "change.*to",
                "update.*to"
            ]
            
            compliance_score = sum(1 for indicator in compliance_indicators if re.search(indicator, content_lower))
            non_compliance_score = sum(1 for indicator in non_compliance_indicators if re.search(indicator, content_lower))
            
            if compliance_score > non_compliance_score and compliance_score > 0:
                action = "KEEP"
            elif non_compliance_score > 0:
                action = "CLEAN"
            else:
                # Default to NEEDS_REVIEW if unclear
                action = "NEEDS_REVIEW"

        def extract(field: str) -> str | None:
            patterns = [
                rf'"(?:{field})"\s*:\s*"([^"]+)"',
                rf"{field}[:\s]+(.+?)(?:\n|[,;]|$)",
            ]
            for p in patterns:
                m = re.search(p, content, re.IGNORECASE)
                if m:
                    val = m.group(1).strip().strip('"').strip()
                    if val and val.lower() not in ("none", "null", "n/a"):
                        return val
            return None

        decision: dict = {"action": action}

        artist = extract("artist")
        if artist:
            decision["artist"] = artist

        title = extract("title")
        if title:
            decision["title"] = title

        track_match = re.search(r'track[_\s]*number[:\s]*(\d+)', content, re.IGNORECASE)
        if track_match:
            decision["track_number"] = int(track_match.group(1))

        confidence = extract("confidence")
        if confidence:
            decision["confidence"] = confidence.upper()

        # Remove any path/filename the LLM might have hallucinated.
        # The agent owns all file paths.
        decision.pop("path", None)
        decision.pop("filename", None)

        reason = extract("reason")
        if action == "KEEP":
            return decision

        return decision

    def _next_track_number(self, riddim: str) -> int:
        """Get the next available track number for a riddim."""
        if riddim not in self._used_track_nums:
            self._used_track_nums[riddim] = set()
            self._track_num_counter[riddim] = 1

        num = self._track_num_counter[riddim]
        self._used_track_nums[riddim].add(num)
        self._track_num_counter[riddim] += 1

        while self._track_num_counter[riddim] in self._used_track_nums[riddim]:
            self._track_num_counter[riddim] += 1

        return num

    # --------------------------------------------------------
    # Approval
    # --------------------------------------------------------

    def show_proposals(self):

        proposals = self.tools.pending_proposals

        if not proposals:
            return []

        print()
        print("=" * 75)
        print("                    PROPOSED CHANGES")
        print("=" * 75)

        for i, p in enumerate(proposals, 1):

            path = Path(p["path"])

            new_name = make_filename(
                p["track_number"],
                p["artist"],
                p["title"],
                path.suffix
            )

            print()
            print(f"[{i}] {path.name}")
            print(f"    - {new_name}")
            print(f"    Artist:     {p['artist']}")
            print(f"    Title:      {p['title']}")
            print(f"    Track:      {p['track_number']}")
            print(f"    Confidence: {p['confidence']}")
            print(f"    Reason:     {p['reason']}")

        print()
        print("=" * 75)

        return proposals

    def apply_proposals(self):

        proposals = self.show_proposals()

        if not proposals:
            return

        if not AUTO_APPLY:
            answer = input(
                "\nApply these changes? [y/N]: "
            ).strip().lower()

            if answer != "y":
                log("Changes rejected by user.")
                return

        if AUTO_APPLY:
            log("AUTO_APPLY enabled - applying without confirmation")

        batch_id = hashlib.sha256(
            str(time.time()).encode()
        ).hexdigest()[:16]

        log(f"Applying batch {batch_id}...")

        for p in proposals:

            path = Path(p["path"])

            try:

                original_metadata = read_metadata(path)

                new_name = make_filename(
                    p["track_number"],
                    p["artist"],
                    p["title"],
                    path.suffix
                )

                new_path = path.parent / new_name

                if new_path.exists() and new_path != path:
                    raise FileExistsError(
                        f"Target already exists: {new_path}"
                    )

                # Jellyfin metadata
                new_metadata = {
                    "title": p["title"],
                    "artist": p["artist"],
                    "album": path.parent.name,
                    "albumartist": "Various Artists",
                    "date": get_riddim_context(path)["year"],
                    "tracknumber": str(p["track_number"]),
                    "genre": "Dancehall",
                }

                self.memory.record_operation(
                    batch_id=batch_id,
                    fingerprint=fingerprint(path),
                    operation="rename_and_tag",
                    original_path=str(path),
                    new_path=str(new_path),
                    original_metadata=original_metadata,
                    new_metadata=new_metadata
                )

                # Write metadata BEFORE rename.
                write_metadata(
                    path,
                    new_metadata
                )

                if new_path != path:
                    path.rename(new_path)

                self.memory.upsert_track({
                    "fingerprint": fingerprint(new_path),
                    "path": str(new_path),
                    "riddim": path.parent.name,
                    "year": new_metadata["date"],
                    "status": "completed",
                    "last_decision": "APPLIED",
                    "confidence": p["confidence"],
                    "artist": p["artist"],
                    "title": p["title"],
                    "album": path.parent.name,
                    "original_path": str(path),
                    "new_path": str(new_path)
                })

                log(
                    f"APPLIED: {path.name} - {new_path.name}",
                    "EXECUTE"
                )

            except Exception as exc:

                error_log(
                    f"Failed to modify {path}: {exc}"
                )

        self.tools.pending_proposals.clear()

        log(
            f"Batch {batch_id} finished."
        )


# ============================================================
# SCAN
# ============================================================

def scan_riddims(root: Path):

    audio_extensions = {
        ".mp3",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".wav",
    }

    # Detect structure:
    #   Case A: root/year/riddim/
    #   Case B: root/riddim/  (flat)

    first_level = sorted(root.iterdir())

    has_year_folders = any(
        p.is_dir()
        and p.name.isdigit()
        and len(p.name) == 4
        for p in first_level
    )

    if has_year_folders:

        for year_folder in first_level:

            if not year_folder.is_dir():
                continue

            if not year_folder.name.isdigit():
                continue

            year = year_folder.name

            for riddim_folder in sorted(
                year_folder.iterdir()
            ):

                if not riddim_folder.is_dir():
                    continue

                yield riddim_folder, year

    else:

        # Flat structure: root/riddim/
        # Try to extract year from folder name.
        for riddim_folder in first_level:

            if not riddim_folder.is_dir():
                continue

            yield riddim_folder, None


# ============================================================
# MAIN
# ============================================================

def main():

    log("RIDDIM JELLYFIN AI AGENT")
    log("=" * 50)

    if not ROOT_FOLDER.exists():
        raise FileNotFoundError(
            f"Root folder doesn't exist:\n{ROOT_FOLDER}"
        )

    log(f"Root: {ROOT_FOLDER}")
    log(f"Model: {MODEL_PATH}")
    log(f"Memory: {STATE_DB}")

    memory = Memory(STATE_DB)

    server = LlamaServer()

    def shutdown(*args):

        log("Shutdown requested.")

        try:
            memory.close()
        except Exception:
            pass

        server.stop()

        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:

        server.start()

        llm = LLM(server)

        agent = RiddimAgent(
            llm,
            memory
        )

        for riddim_folder, year in scan_riddims(
            ROOT_FOLDER
        ):

                log("")
                log("=" * 70)
                log(
                    f"RIDDIM: {riddim_folder.name}"
                )
                log(
                    f"YEAR: {year}"
                )
                log("=" * 70)

                tracks = list_riddim_tracks(
                    riddim_folder
                )

                log(
                    f"{len(tracks)} tracks found."
                )

                # --------------------------------------------------------
                # BATCH-LEVEL PLANNING
                # --------------------------------------------------------

                planner = RiddimPlanner(
                    llm,
                    memory
                )

                plan = planner.create_plan(
                    folder=riddim_folder,
                    riddim=riddim_folder.name,
                    year=year,
                    tracks=[
                        {
                            "path": str(
                                riddim_folder / x["filename"]
                            ),
                            "filename": x["filename"],
                            "metadata": x["metadata"]
                        }
                        for x in tracks
                    ]
                )

                # --------------------------------------------------------
                # DISPLAY PLAN
                # --------------------------------------------------------

                print()
                print("BATCH PLAN")
                print("-" * 70)

                for decision in plan["tracks"]:

                    print(
                        f'{decision["action"]:15} '
                        f'{decision.get("filename", decision["path"])}'
                    )

                    if decision.get("artist"):
                        print(
                            f'    - {decision["artist"]} - '
                            f'{decision["title"]}'
                        )

                    print(
                        f'    confidence: '
                        f'{decision["confidence"]}'
                    )

                    print(
                        f'    reason: '
                        f'{decision["reason"]}'
                    )

                if plan.get("batch_warnings"):

                    print()
                    print("BATCH WARNINGS")

                    for warning in plan["batch_warnings"]:
                        print(f"    ! {warning}")

                # --------------------------------------------------------
                # TRACK AGENT - PROCESS EVERY TRACK
                # --------------------------------------------------------

                for decision in plan["tracks"]:

                    path = Path(decision["path"])
                    assert path.exists(), f"Path does not exist: {path}"

                    if decision["action"] == "NEEDS_REVIEW":

                        ai_log(
                            f"Skipping uncertain track: {path.name}"
                        )

                        continue

                    if decision["action"] == "KEEP":

                        log(
                            f"Processing already-correct track: {path.name}"
                        )

                    agent.process_track(
                        path=path,
                        riddim=riddim_folder.name,
                        year=year
                    )

                # --------------------------------------------------------
                # HUMAN APPROVAL
                # --------------------------------------------------------

                if agent.tools.pending_proposals:
                    agent.apply_proposals()

        log("")
        log("=" * 70)
        log("COLLECTION PROCESSING COMPLETE")
        log("=" * 70)

    except KeyboardInterrupt:

        log("Interrupted by user.")

    except Exception as exc:

        error_log(str(exc))

    finally:

        memory.close()
        server.stop()


if __name__ == "__main__":
    main()