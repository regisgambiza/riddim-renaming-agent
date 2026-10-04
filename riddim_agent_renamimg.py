import os
import sys
import json
import re
import time
import sqlite3
import hashlib
import signal
import subprocess
import traceback
from pathlib import Path
from typing import Any
from datetime import datetime
import yaml

import requests
from mutagen import File as MutagenFile
from tqdm import tqdm


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"C:\Users\regis\.lmstudio\models\lmstudio-community\Qwen3.6-35B-A3B-GGUF\Qwen3.6-35B-A3B-Q4_K_M.gguf"

# Change this if llama-server.exe is somewhere else.
LLAMA_SERVER_EXE = r"C:\Tools\llama.cpp\llama-server.exe"

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8099

# Your riddim collection
ROOT_FOLDER = Path(r"D:\Riddim_Juggling")

# Persistent agent memory
STATE_DB = Path("riddim_agent_memory.db")

# Default = no changes until approved
AUTO_APPLY = True

# Number of agent iterations before forcing a fresh context
MAX_AGENT_STEPS = 40

# Seconds to pause between scanned folders during initial scan.
# Increase this to reduce I/O load on a slow/failing HDD.
SCAN_DELAY_SECONDS = 0.0

# AI parameters
TEMPERATURE = 0.1
MAX_TOKENS = 1500

# llama.cpp settings
CONTEXT_SIZE = 32768
N_GPU_LAYERS = 20
CPU_THREADS = 12
BATCH_SIZE = 4096
UBATCH_SIZE = 1024
FLASH_ATTENTION = 1


def _load_rules():
    """Load strict rules from YAML config file.
    Returns a list of rules sorted by priority (highest first).
    If the config is missing or invalid, returns an empty list.
    """
    rules_path = Path(__file__).parent / "config" / "rules.yaml"
    try:
        with open(rules_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        rules = config.get("strict_rules", [])
        rules.sort(key=lambda r: r.get("priority", 100), reverse=True)
        return rules
    except Exception:
        print(f"[WARNING] Could not load rules from {rules_path}. Using normal AI reasoning.", flush=True)
        return []


RULES = _load_rules()


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


def error_log_detailed(message: str, exc_info=None):
    """Log error with full traceback and context."""
    lines = [
        f"ERROR: {message}",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if exc_info:
        lines.append("Traceback:")
        lines.extend(traceback.format_tb(exc_info[2]))
    else:
        lines.append("Traceback: (no exception info captured)")
    lines.append("=" * 80)
    for line in lines:
        log(line, "DETAILED_ERROR")


# ============================================================
# PROCESSING LOG
# ============================================================

class ProcessingLog:
    """Handles detailed per-track logging of metadata changes."""

    def __init__(self, log_dir: Path = None):
        if log_dir is None:
            log_dir = Path(__file__).parent / "logs"
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = log_dir / f"processing_log_{timestamp}.txt"
        self._write_header()

    def _write_header(self):
        """Initialize log file with header."""
        header = (
            f"RIDDIM JELLYFIN AI AGENT - PROCESSING LOG\n"
            f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Log file: {self.log_path}\n"
            f"{'='*80}\n\n"
        )
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(header)

    def log_track(self, path: Path, before: dict, after: dict, status: str,
                  reason: str = "", rule_applied: str = "", decision: str = "",
                  track_number_before: int = None, track_number_after: int = None):
        """Log a single track's before/after metadata and processing info."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"==================================================\n")
            f.write(f"TRACK: {path.name}\n")
            f.write(f"PATH: {path}\n")
            f.write(f"STATUS: {status}\n")
            f.write(f"TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"DECISION: {decision}\n")
            f.write(f"REASON: {reason}\n")
            f.write(f"RULE APPLIED: {rule_applied}\n")
            f.write(f"TRACK NUMBER: {track_number_before or 0} -> {track_number_after or 0}\n")
            f.write(f"--------------------------------------------------\n")
            f.write(f"BEFORE:                                  AFTER:\n")
            # Order of fields for display
            fields = [
                ("title", "Title"),
                ("artist", "Artist"),
                ("album", "Album"),
                ("albumartist", "Album Artist"),
                ("tracknumber", "Track Number"),
                ("date", "Year"),
                ("genre", "Genre"),
                ("compilation", "Compilation"),
                ("discnumber", "Disc Number"),
            ]
            for key, label in fields:
                # Handle missing/empty values
                before_val = self._format_value(before.get(key))
                after_val = self._format_value(after.get(key))
                f.write(f"{label:<20} = {before_val:<25} {label:<20} = {after_val}\n")
            # Show file rename if changed
            before_path = before.get("path") or str(path)
            after_path = after.get("path") or str(path)
            if before_path != after_path:
                f.write(f"FILE NAME    = {Path(before_path).name:<25} FILE NAME    = {Path(after_path).name}\n")
            f.write(f"==================================================\n\n")

    def log_error(self, message: str, exc_info=None):
        """Log detailed error with traceback to the processing log."""
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"ERROR: {message}\n")
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            if exc_info:
                f.write(f"Traceback:\n")
                f.write("".join(traceback.format_tb(exc_info[2])))
            else:
                f.write(f"Traceback: (no exception info captured)\n")
            f.write(f"{'='*80}\n\n")

    def _format_value(self, val):
        """Format value for display."""
        if val is None or val == "":
            return "None"
        # If it's a list, take first element
        if isinstance(val, list):
            val = val[0] if val else None
        if val is None:
            return "None"
        return str(val).strip()


# Global log instance (set in main)
processing_log = None


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
            reason TEXT,
            artist TEXT,
            title TEXT,
            album TEXT,
            original_path TEXT,
            new_path TEXT,
            artist_conf TEXT,
            title_conf TEXT,
            album_conf TEXT,
            albumartist_conf TEXT,
            year_conf TEXT,
            tracknumber_conf TEXT,
            genre_conf TEXT,
            compilation_conf TEXT,
            discnumber_conf TEXT,
            fields_changed TEXT,
            fields_skipped TEXT,
            fields_skipped_reason TEXT,
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

         CREATE TABLE IF NOT EXISTS folder_status (
             folder_path TEXT PRIMARY KEY,
             riddim TEXT NOT NULL,
             year TEXT,
             total_tracks INTEGER DEFAULT 0,
             completed_tracks INTEGER DEFAULT 0,
             failed_tracks INTEGER DEFAULT 0,
             needs_review_tracks INTEGER DEFAULT 0,
             status TEXT NOT NULL DEFAULT 'pending',
             last_scan TIMESTAMP,
             last_processed TIMESTAMP,
             error_count INTEGER DEFAULT 0
         );

         CREATE TABLE IF NOT EXISTS artist_cache (
             artist_name TEXT PRIMARY KEY,
             verified INTEGER NOT NULL DEFAULT 0,  -- 0 = unknown, 1 = verified real, 2 = verified fake
             spotify_id TEXT,
             youtube_channel_id TEXT,
             popularity INTEGER,
             followers INTEGER,
             genres TEXT,  -- JSON array
             last_verified TIMESTAMP,
             updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
         );
         """)
        self.conn.commit()

        columns = [row[1] for row in self.conn.execute("PRAGMA table_info(tracks)")]
        for col in [
            "reason",
            "artist_conf",
            "title_conf",
            "album_conf",
            "albumartist_conf",
            "year_conf",
            "tracknumber_conf",
            "genre_conf",
            "compilation_conf",
            "discnumber_conf",
            "fields_changed",
            "fields_skipped",
            "fields_skipped_reason",
        ]:
            if col not in columns:
                self.conn.execute(f"ALTER TABLE tracks ADD COLUMN {col} TEXT")
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
            last_decision, confidence, reason, artist, title,
            album, original_path, new_path,
            artist_conf, title_conf, album_conf, albumartist_conf,
            year_conf, tracknumber_conf, genre_conf, compilation_conf,
            discnumber_conf, fields_changed, fields_skipped,
            fields_skipped_reason
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(fingerprint) DO UPDATE SET
            path=excluded.path,
            riddim=excluded.riddim,
            year=excluded.year,
            status=excluded.status,
            last_decision=excluded.last_decision,
            confidence=excluded.confidence,
            reason=excluded.reason,
            artist=excluded.artist,
            title=excluded.title,
            album=excluded.album,
            original_path=excluded.original_path,
            new_path=excluded.new_path,
            artist_conf=excluded.artist_conf,
            title_conf=excluded.title_conf,
            album_conf=excluded.album_conf,
            albumartist_conf=excluded.albumartist_conf,
            year_conf=excluded.year_conf,
            tracknumber_conf=excluded.tracknumber_conf,
            genre_conf=excluded.genre_conf,
            compilation_conf=excluded.compilation_conf,
            discnumber_conf=excluded.discnumber_conf,
            fields_changed=excluded.fields_changed,
            fields_skipped=excluded.fields_skipped,
            fields_skipped_reason=excluded.fields_skipped_reason,
            updated_at=CURRENT_TIMESTAMP
        """, (
            data["fingerprint"],
            data["path"],
            data.get("riddim"),
            data.get("year"),
            data.get("status", "pending"),
            data.get("last_decision"),
            data.get("confidence"),
            data.get("reason"),
            data.get("artist"),
            data.get("title"),
            data.get("album"),
            data.get("original_path"),
            data.get("new_path"),
            data.get("artist_conf"),
            data.get("title_conf"),
            data.get("album_conf"),
            data.get("albumartist_conf"),
            data.get("year_conf"),
            data.get("tracknumber_conf"),
            data.get("genre_conf"),
            data.get("compilation_conf"),
            data.get("discnumber_conf"),
            data.get("fields_changed"),
            data.get("fields_skipped"),
            data.get("fields_skipped_reason"),
        ))
        self._drop_stale_track_rows(data)
        self.conn.commit()

    def _drop_stale_track_rows(self, data):
        """Drop older rows for the same file once it is finished.

        A metadata write changes size and mtime, so the old fingerprint
        no longer matches the file. Those leftover rows made the next
        launch treat a finished track as new.
        """
        if data.get("status") not in {"completed", "approved", "compliant"}:
            return
        paths = []
        for key in ("path", "original_path", "new_path"):
            value = data.get(key)
            if value and value not in paths:
                paths.append(value)
        for stored_path in paths:
            self.conn.execute(
                """DELETE FROM tracks
                WHERE fingerprint != ?
                  AND (path = ? OR original_path = ? OR new_path = ?)""",
                (data["fingerprint"], stored_path, stored_path, stored_path)
            )

    def track_status(self, path: Path) -> str | None:
        """Best known status for this file, ignoring mtime.

        Finished wins over an older pending row for the same path.
        """
        values = []
        raw = str(path)
        for candidate in (raw, os.path.normpath(raw)):
            if candidate not in values:
                values.append(candidate)
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = None
        if resolved and resolved not in values:
            values.append(resolved)

        clauses = []
        params = []
        for value in values:
            clauses.append("path = ? OR original_path = ? OR new_path = ?")
            params.extend((value, value, value))

        rows = self.conn.execute(
            f"SELECT status FROM tracks WHERE {' OR '.join(clauses)}",
            params,
        ).fetchall()
        if not rows:
            return None

        rank = {
            "completed": 0,
            "approved": 0,
            "compliant": 0,
            "needs_review": 1,
            "error": 2,
            "pending_review": 3,
            "pending": 4,
        }
        return min((row["status"] for row in rows), key=lambda status: rank.get(status, 9))

    def track_is_done(self, path: Path) -> bool:
        return self.track_status(path) in {"completed", "approved", "compliant"}

    def remember_completed_track(self, path: Path, riddim: str, year: str | None, reason: str):
        if not path.exists():
            return
        resolved = str(path.resolve())
        self.upsert_track({
            "fingerprint": fingerprint(path),
            "path": resolved,
            "riddim": riddim,
            "year": year,
            "status": "completed",
            "last_decision": "SKIP",
            "confidence": "HIGH",
            "reason": reason,
            "original_path": resolved,
            "new_path": resolved,
        })

    def update_folder_track_count(self, folder_path: Path, total_tracks: int):
        self.conn.execute(
            """UPDATE folder_status
            SET total_tracks = ?,
                last_scan = CURRENT_TIMESTAMP
            WHERE folder_path = ?""",
            (total_tracks, str(folder_path.resolve()))
        )
        self.conn.commit()

    def get_folder_status(self, riddim: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM folder_status WHERE riddim=? ORDER BY last_scan DESC LIMIT 1",
            (riddim,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_folder_status(self, folder_path: Path, riddim: str, year: str | None = None):
        """Insert a folder row, or refresh its scan time.

        An existing row keeps its status and counters. Replacing the row
        here used to reset an interrupted folder back to pending.
        """
        self.conn.execute(
            """INSERT INTO folder_status
            (folder_path, riddim, year, total_tracks, completed_tracks, failed_tracks,
             needs_review_tracks, status, last_scan, last_processed, error_count)
            VALUES (?, ?, ?, 0, 0, 0, 0, 'pending', CURRENT_TIMESTAMP, NULL, 0)
            ON CONFLICT(folder_path) DO UPDATE SET
                riddim=excluded.riddim,
                year=COALESCE(excluded.year, folder_status.year),
                last_scan=CURRENT_TIMESTAMP
            """,
            (str(folder_path.resolve()), riddim, year)
        )
        self.conn.commit()

    def update_folder_progress(self, folder_path: Path, riddim: str,
                               completed: bool = False, failed: bool = False,
                               needs_review: bool = False):
        now = "CURRENT_TIMESTAMP"
        if completed:
            self.conn.execute(
                """UPDATE folder_status
                SET completed_tracks = completed_tracks + 1,
                    status = CASE WHEN completed_tracks + 1 >= total_tracks THEN 'completed' ELSE status END,
                    last_processed = CURRENT_TIMESTAMP
                WHERE folder_path = ?""",
                (str(folder_path.resolve()),)
            )
        if failed:
            self.conn.execute(
                """UPDATE folder_status
                SET failed_tracks = failed_tracks + 1,
                    status = CASE WHEN status != 'failed' THEN 'partial' ELSE status END,
                    error_count = error_count + 1
                WHERE folder_path = ?""",
                (str(folder_path.resolve()),)
            )
        if needs_review:
            self.conn.execute(
                """UPDATE folder_status
                SET needs_review_tracks = needs_review_tracks + 1,
                    status = CASE WHEN status != 'needs_review' THEN 'needs_review' ELSE status END
                WHERE folder_path = ?""",
                (str(folder_path.resolve()),)
            )
        self.conn.commit()

    def mark_folder_complete(self, folder_path: Path, riddim: str):
        self.conn.execute(
            """UPDATE folder_status
            SET status = 'completed',
                last_processed = CURRENT_TIMESTAMP
            WHERE folder_path = ?""",
            (str(folder_path.resolve()),)
        )
        self.conn.commit()

    def get_folder_progress(self, folder_path: Path) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM folder_status WHERE folder_path=?",
            (str(Path(folder_path).resolve()),)
        ).fetchone()
        return dict(row) if row else None

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

    def get_unresolved_tracks(self, riddim: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM tracks WHERE riddim=? AND status NOT IN "
            "('completed', 'approved', 'compliant', 'pending_review')",
            (riddim,)
        ).fetchall()

        return [dict(r) for r in rows]

    # --- Artist Cache Methods ---
    def get_artist_cache(self, artist_name: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM artist_cache WHERE artist_name=?",
            (artist_name,)
        ).fetchone()
        return dict(row) if row else None

    def upsert_artist_cache(self, artist_name: str, verified: int = 0,
                             spotify_id: str = None, youtube_channel_id: str = None,
                             popularity: int = None, followers: int = None,
                             genres: list = None):
        import json
        genres_json = json.dumps(genres) if genres else None
        
        # Build dynamic UPDATE clause to only update provided fields
        updates = ["verified=excluded.verified"]
        params = [artist_name, verified, spotify_id, youtube_channel_id, 
                  popularity, followers, genres_json]
        
        if spotify_id is not None:
            updates.append("spotify_id=excluded.spotify_id")
        if youtube_channel_id is not None:
            updates.append("youtube_channel_id=excluded.youtube_channel_id")
        if popularity is not None:
            updates.append("popularity=excluded.popularity")
        if followers is not None:
            updates.append("followers=excluded.followers")
        if genres_json is not None:
            updates.append("genres=excluded.genres")
            
        update_clause = ", ".join(updates)
        
        self.conn.execute(f"""
        INSERT INTO artist_cache (artist_name, verified, spotify_id, youtube_channel_id,
                                  popularity, followers, genres, last_verified)
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT(artist_name) DO UPDATE SET
            {update_clause},
            last_verified=CURRENT_TIMESTAMP
        """, params)
        self.conn.commit()

    def get_or_create_artist_cache(self, artist_name: str) -> dict:
        cached = self.get_artist_cache(artist_name)
        if cached is None:
            cached = {
                "artist_name": artist_name,
                "verified": 0,
                "spotify_id": None,
                "youtube_channel_id": None,
                "popularity": None,
                "followers": None,
                "genres": None,
                "last_verified": None
            }
        return cached

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
    try:
        audio = MutagenFile(str(path), easy=True)
    except Exception as exc:
        return {
            "error": f"Unable to read audio metadata: {exc}"
        }

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
        "compilation": first("compilation"),
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
        "compilation": metadata.get("compilation"),
    }

    for key, value in mapping.items():
        if value is not None:
            audio[key] = [str(value)]

    audio.save()
    return True


def evaluate_field_confidence(field_name: str, current_value, proposed_value, evidence: dict | None = None):
    """Return (confidence, reason) for a single metadata field."""
    evidence = evidence or {}
    sources = evidence.get("sources", {}) if isinstance(evidence, dict) else {}

    if proposed_value is None:
        return ("SKIP", "No reliable value available for this field.")

    if current_value is not None and str(current_value).strip() == str(proposed_value).strip():
        if sources:
            return ("HIGH", "Existing value matches the available evidence.")
        return ("MEDIUM", "Value already matches the best local evidence.")

    if sources:
        normalized_sources = [str(v).strip() for v in sources.values() if v not in (None, "", [], {})]
        if any(str(proposed_value).strip() == str(src) for src in normalized_sources):
            return ("HIGH", f"Value is supported by {len(normalized_sources)} local evidence source(s).")

    if field_name in {"artist", "title", "album", "albumartist", "year", "tracknumber", "discnumber"}:
        return ("MEDIUM", "Reasonable local evidence for this field, but not fully confirmed.")

    if field_name == "genre":
        return ("LOW", "Genre cannot be inferred confidently without stronger evidence.")

    return ("LOW", "Insufficient evidence for a reliable metadata change.")


def investigate_metadata(path: Path, riddim: str | None = None, year: str | None = None, sibling_tracks: list[dict] | None = None):
    """Evaluate all important embedded fields for a track without modifying the file."""
    file_path = Path(path)
    metadata = read_metadata(file_path)
    if "error" in metadata:
        return {
            "path": str(file_path),
            "status": "error",
            "decision": "NEEDS_REVIEW",
            "field_decisions": {},
            "reason": metadata.get("error"),
            "metadata": metadata,
            "proposed_changes": {},
        }

    filename_info = RiddimAgent._parse_filename(file_path)
    context = get_riddim_context(file_path)
    folder_name = riddim or file_path.parent.name or context.get("riddim")
    year_value = year or context.get("year")

    artist = strip_leading_track_number(metadata.get("artist")) or filename_info.get("artist") or ""
    title = (metadata.get("title") or "").strip() or filename_info.get("title") or ""
    album = (metadata.get("album") or folder_name or "").strip()
    albumartist = (metadata.get("albumartist") or "Various Artists").strip() or "Various Artists"
    date_value = metadata.get("date") or year_value
    tracknumber = metadata.get("tracknumber")
    try:
        if isinstance(tracknumber, list):
            tracknumber = tracknumber[0] if tracknumber else None
        if tracknumber is not None:
            tracknumber = int(str(tracknumber).split("/")[0].strip())
    except Exception:
        tracknumber = None
    if tracknumber is None and filename_info.get("track_number") is not None:
        tracknumber = filename_info.get("track_number")
    genre = metadata.get("genre")
    compilation = metadata.get("compilation") or "Yes"
    discnumber = metadata.get("discnumber") or "1"

    evidence = {
        "sources": {
            "filename_artist": filename_info.get("artist"),
            "metadata_artist": metadata.get("artist"),
            "filename_title": filename_info.get("title"),
            "metadata_title": metadata.get("title"),
            "riddim": folder_name,
            "year": year_value,
            "tracknumber": tracknumber,
            "albumartist": albumartist,
        }
    }

    field_decisions = {}
    for field_name, current_value, proposed_value in [
        ("artist", metadata.get("artist"), artist or None),
        ("title", metadata.get("title"), title or None),
        ("album", metadata.get("album"), album or None),
        ("albumartist", metadata.get("albumartist"), albumartist or None),
        ("year", metadata.get("date"), date_value),
        ("tracknumber", metadata.get("tracknumber"), tracknumber),
        ("genre", metadata.get("genre"), genre),
        ("compilation", metadata.get("compilation"), compilation),
        ("discnumber", metadata.get("discnumber"), discnumber),
    ]:
        confidence, reason = evaluate_field_confidence(field_name, current_value, proposed_value, evidence)
        if proposed_value is None:
            field_decisions[field_name] = {
                "current": current_value,
                "proposed": None,
                "decision": "SKIP",
                "confidence": confidence,
                "reason": reason,
            }
        elif current_value is not None and str(current_value).strip() == str(proposed_value).strip():
            field_decisions[field_name] = {
                "current": current_value,
                "proposed": proposed_value,
                "decision": "KEEP",
                "confidence": confidence,
                "reason": reason,
            }
        else:
            field_decisions[field_name] = {
                "current": current_value,
                "proposed": proposed_value,
                "decision": "METADATA_FIX",
                "confidence": confidence,
                "reason": reason,
            }

    proposed_changes = {
        field: details["proposed"]
        for field, details in field_decisions.items()
        if details["proposed"] is not None and details["decision"] != "KEEP"
    }

    status = "compliant" if all(details["decision"] in {"KEEP", "SKIP"} for details in field_decisions.values()) else "needs_review"
    if any(details["decision"] == "METADATA_FIX" and details["confidence"] == "LOW" for details in field_decisions.values()):
        status = "needs_review"

    return {
        "path": str(file_path),
        "status": status,
        "decision": "METADATA_CHECK",
        "metadata": metadata,
        "field_decisions": field_decisions,
        "proposed_changes": proposed_changes,
        "reason": "Embedded metadata evaluated against local evidence.",
    }


def verify_metadata_after_write(path: Path, expected: dict | None = None):
    """Re-read the file after a write and verify the metadata fields that matter."""
    file_path = Path(path)
    actual = read_metadata(file_path)
    if "error" in actual:
        return {
            "path": str(file_path),
            "verified": False,
            "actual": actual,
            "missing": [],
            "errors": [actual.get("error")],
        }

    expected = expected or {}
    missing = []
    errors = []
    verified = True

    for field_name, expected_value in expected.items():
        actual_value = actual.get(field_name)
        if actual_value is None and expected_value is not None:
            missing.append(field_name)
            verified = False
            errors.append(f"Missing field after write: {field_name}")
            continue
        if expected_value is not None and str(actual_value).strip() != str(expected_value).strip():
            verified = False
            errors.append(f"Field mismatch: {field_name}={actual_value!r} != {expected_value!r}")

    return {
        "path": str(file_path),
        "verified": verified,
        "actual": actual,
        "missing": missing,
        "errors": errors,
    }


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


def strip_leading_track_number(text: str | None) -> str:
    value = (text or "").strip()

    while True:
        match = re.match(r'^(\d{1,2})(?:\s*[.-]\s*|\s+)(.+)$', value)
        if not match:
            return value
        value = match.group(2).strip()


def make_filename(track_number, artist, title, extension):
    if not artist or not title:
        raise ValueError(
            f"Artist and title must be non-empty: "
            f"artist={artist!r}, title={title!r}"
        )

    artist = safe_filename(artist)
    title = safe_filename(title)

    return f"{artist} - {title}{extension}"


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
            "name": "investigate_metadata",
            "description": "Evaluate embedded metadata for a track using file, folder, and sibling evidence without changing the file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "riddim": {"type": "string"},
                    "year": {"type": "string"},
                    "sibling_tracks": {"type": "array"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_metadata_after_write",
            "description": "Re-read a track after writing metadata and verify that the expected tags match the file on disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "expected": {"type": "object"}
                },
                "required": ["path"]
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
    },
    {
        "type": "function",
        "function": {
            "name": "search_artist_online",
            "description": "Search for an artist on Spotify and YouTube to verify if they are real. Uses cached results when available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "artist_name": {
                        "type": "string",
                        "description": "The name of the artist to verify"
                    }
                },
                "required": ["artist_name"]
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

        if name == "investigate_metadata":
            result = investigate_metadata(
                Path(args["path"]),
                riddim=args.get("riddim"),
                year=args.get("year"),
                sibling_tracks=args.get("sibling_tracks"),
            )
            result_log(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
            return result

        if name == "verify_metadata_after_write":
            result = verify_metadata_after_write(
                Path(args["path"]),
                expected=args.get("expected")
            )
            result_log(json.dumps(result, ensure_ascii=False, indent=2)[:2000])
            return result

        if name == "search_artist_online":
            return self._search_artist_online(args)

        raise ValueError(f"Unknown tool: {name}")

    def _search_artist_online(self, args: dict) -> dict:
        """Search Spotify and YouTube for artist verification."""
        artist_name = args.get("artist_name", "").strip()
        if not artist_name:
            return {"error": "No artist name provided"}

        # Check cache first - if we have BOTH IDs, return from cache immediately
        cached = self.memory.get_artist_cache(artist_name)
        if cached and cached.get("verified") in (1, 2):
            # If both sources verified, return from cache
            if cached.get("spotify_id") and cached.get("youtube_channel_id"):
                return {
                    "artist_name": artist_name,
                    "verified": cached["verified"],
                    "source": "cache",
                    "spotify_id": cached.get("spotify_id"),
                    "youtube_channel_id": cached.get("youtube_channel_id"),
                    "genres": cached.get("genres"),
                    "popularity": cached.get("popularity"),
                    "followers": cached.get("followers")
                }

        # Need to search - start with cached values if any
        has_spotify = bool(cached.get("spotify_id")) if cached else False
        has_youtube = bool(cached.get("youtube_channel_id")) if cached else False

        # Fallback if not fully cached
        if not has_spotify or not has_youtube:
            if cached:
                result = {
                    "artist_name": artist_name,
                    "verified": cached.get("verified", 0),
                    "source": "api",
                    "spotify_id": cached.get("spotify_id"),
                    "youtube_channel_id": cached.get("youtube_channel_id"),
                    "genres": cached.get("genres"),
                    "popularity": cached.get("popularity"),
                    "followers": cached.get("followers")
                }
            else:
                result = {"artist_name": artist_name, "verified": 0, "source": "api"}
        else:
            result = {"artist_name": artist_name, "verified": 0, "source": "api"}

        # Search Spotify if needed
        if not has_spotify:
            try:
                client_id = "072209652b0243f3b6f357577e103adf"
                client_secret = "2a868134eeab4a9db5a43e4fb3ce610c"
                import base64
                auth_str = f"{client_id}:{client_secret}"
                b64auth = base64.b64encode(auth_str.encode()).decode()
                headers = {"Authorization": f"Basic {b64auth}"}
                token_resp = requests.post(
                    "https://accounts.spotify.com/api/token",
                    headers=headers,
                    data={"grant_type": "client_credentials"},
                    timeout=10
                )
                if token_resp.status_code == 200:
                    access_token = token_resp.json().get("access_token")
                    if access_token:
                        search_headers = {"Authorization": f"Bearer {access_token}"}
                        search_resp = requests.get(
                            "https://api.spotify.com/v1/search",
                            headers=search_headers,
                            params={"q": artist_name, "type": "artist", "limit": 1},
                            timeout=10
                        )
                        if search_resp.status_code == 200:
                            artists = search_resp.json().get("artists", {}).get("items", [])
                            if artists:
                                artist_data = artists[0]
                                spotify_id = artist_data.get("id")
                                name = artist_data.get("name")
                                popularity = artist_data.get("popularity", 0)
                                genres = artist_data.get("genres", [])
                                if result.get("verified") == 0:
                                    result["verified"] = 1
                                result["spotify_id"] = spotify_id
                                result["name"] = name
                                result["popularity"] = popularity
                                result["genres"] = genres
                                self.memory.upsert_artist_cache(
                                    artist_name,
                                    verified=result["verified"],
                                    spotify_id=spotify_id,
                                    popularity=popularity,
                                    genres=genres
                                )
            except Exception as e:
                result["spotify_error"] = str(e)

        # Search YouTube if needed
        if not has_youtube:
            youtube_key = os.environ.get("YOUTUBE_API_KEY")
            if youtube_key:
                try:
                    yt_resp = requests.get(
                        "https://www.googleapis.com/youtube/v3/search",
                        params={
                            "part": "snippet",
                            "q": artist_name,
                            "type": "channel",
                            "maxResults": 1,
                            "key": youtube_key
                        },
                        timeout=10
                    )
                    if yt_resp.status_code == 200:
                        items = yt_resp.json().get("items", [])
                        if items:
                            channel = items[0]
                            youtube_id = channel.get("id", {}).get("channelId")
                            if result.get("verified") == 0:
                                result["verified"] = 1
                            result["youtube_channel_id"] = youtube_id
                            result["youtube_title"] = channel.get("snippet", {}).get("title")
                            result["youtube_description"] = channel.get("snippet", {}).get("description")
                            self.memory.upsert_artist_cache(
                                artist_name,
                                verified=2 if result["verified"] > 1 else 1,
                                youtube_channel_id=youtube_id,
                                genres=result.get("genres")
                            )
                except Exception as e:
                    result["youtube_error"] = str(e)
            else:
                result["youtube_note"] = "Set YOUTUBE_API_KEY env var for YouTube search"
        else:
            result["verified"] = 2

        # If nothing verified from API, mark as 0 (unknown/fake)
        if result.get("verified") == 0:
            self.memory.upsert_artist_cache(artist_name, verified=0)

        return result

    def verify_artist(self, artist_name: str, riddim_name: str) -> dict:
        """
        Mandatory artist verification.
        1. Check local cache
        2. If not cached or unverified → search Spotify+YouTube
        3. Return {"name": verified_name, "verified": 1|2, "source": "cache"|"api"|"none"}
        """
        if not artist_name or not artist_name.strip():
            return {"name": "", "verified": 0, "source": "none", "reason": "Empty artist name"}

        # 1. Check cache
        cached = self.memory.get_artist_cache(artist_name)
        if cached and cached.get("verified") in (1, 2):
            return {
                "name": artist_name,
                "verified": cached["verified"],
                "source": cached.get("source", "cache"),
                "spotify_id": cached.get("spotify_id"),
                "youtube_channel_id": cached.get("youtube_channel_id"),
            }

        # 2. Search (both Spotify and YouTube via _search_artist_online)
        result = self._search_artist_online({"artist_name": artist_name})
        verified = result.get("verified", 0)

        if verified in (1, 2):
            return {
                "name": result.get("artist_name", artist_name),
                "verified": verified,
                "source": result.get("source", "api"),
                "spotify_id": result.get("spotify_id"),
                "youtube_channel_id": result.get("youtube_channel_id"),
            }

        # 3. Still unverified after search
        return {
            "name": artist_name,
            "verified": 0,
            "source": "none",
            "reason": "Artist not found on Spotify or YouTube",
        }


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
                "-t", str(CPU_THREADS),
                "-b", str(BATCH_SIZE),
                "-ub", str(UBATCH_SIZE),
                "-fa", str(FLASH_ATTENTION),
                "--poll", "50",
                "--mmap",
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
        self._cache = {}

    def chat(self, messages, tools=None, response_format=None):

        cache_key = json.dumps(messages, sort_keys=True, ensure_ascii=False)
        if cache_key in self._cache:
            return self._cache[cache_key]

        payload = {
            "model": "local-model",
            "messages": messages,
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        response = requests.post(
            f"{self.server.base_url}/v1/chat/completions",
            json=payload,
            timeout=1800
        )
        
        if not response.ok:
            error_log(f"LLM API error: {response.status_code} {response.reason}")
            error_log(f"Request URL: {self.server.base_url}/v1/chat/completions")
            error_log(f"Request payload keys: {list(payload.keys())}")
            error_log(f"Response text: {response.text[:1000]}")
        
        response.raise_for_status()

        data = response.json()
        message = data["choices"][0]["message"]

        # Handle models that put output in reasoning_content instead of content
        content = message.get("content", "")
        if not content and message.get("reasoning_content"):
            content = message["reasoning_content"]
            message["content"] = content

        self._cache[cache_key] = message
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
            f"PLANNER: analyzing riddim batch: {riddim} ({len(tracks)} tracks)"
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

            for key in ("title", "artist", "album", "albumartist", "genre", "date", "tracknumber", "discnumber", "compilation"):
                value = metadata.get(key)
                if value not in (None, "", [], {}):
                    compact_metadata[key] = value

            compact_tracks.append({
                "path": track.get("path"),
                "filename": track.get("filename"),
                "metadata": compact_metadata,
            })

        # Build artist verification context: cache lookups for each track's artist
        artist_context = []
        for track in compact_tracks:
            artist_name = track["metadata"].get("artist", "") or track["metadata"].get("title", "")
            if artist_name:
                cached = self.memory.get_artist_cache(artist_name) if self.memory else None
                artist_context.append({
                    "artist_name": artist_name,
                    "verified": cached.get("verified", 0) if cached else 0,
                    "source": cached.get("source", "none") if cached else "none",
                    "spotify_id": cached.get("spotify_id") if cached else None,
                    "youtube_channel_id": cached.get("youtube_channel_id") if cached else None,
                })

        plan = {
            "riddim": riddim,
            "year": year,
            "summary": (
                f"Analyzed {len(compact_tracks)} tracks in batch "
                f"with artist/title identification."
            ),
            "strategy": "Batch planning with artist verification context.",
            "tracks": [],
            "batch_warnings": [],
        }

        messages = [
            {
                "role": "system",
                "content": """
You are the batch planner for a Jellyfin music-cleanup agent.
Your job is to analyze ALL tracks in a riddim folder and identify the real ARTIST and TITLE for each track.

You receive:
- riddim folder name
- year folder
- list of tracks with: filename, embedded metadata, and cached artist verification status

JELLYFIN FILENAME STANDARD (MUST follow exactly):
- Format: Artist - Title.ext
- Example: "Capleton - In Her Heart.mp3"
- Any leading track number in the filename is WRONG and must be removed.
- Examples of WRONG filenames: "01 - Capleton - In Her Heart.mp3", "01. Capleton - In Her Heart.mp3", "02 - 02 Assassin - Still Want More.mp3"

ARTIST/TITLE IDENTIFICATION RULES:
- The filename may use patterns like "Artist - Title", "Artist @ Title", "Artist # Title", "NN - Artist - Title"
- The metadata may have artist/title swapped with the riddim name (e.g., metadata artist="Riddim Name", title="Real Artist")
- If metadata artist matches the riddim folder name, it is LIKELY SWAPPED — the real artist is in the title field
- If filename pattern is "Riddim @ Artist", the riddim is the folder name, not the artist
- Use sibling tracks for consistency: if multiple tracks show the same artist name, it is likely correct
- Cached artist verification: if provided, it shows whether an artist name was verified via Spotify/YouTube
  - verified: 0 = not verified, 1 = Spotify verified, 2 = Spotify+YouTube verified
  - source: "cache" (from local DB), "api" (from live search), "none" (no verification attempted)

SWAP DETECTION:
- If you detect a swap, set "action": "SWAP_ARTIST_TITLE" and the agent will swap artist/title before writing metadata
- After swap, re-evaluate whether the new artist is verified or needs review

ACTIONS:
- KEEP: track already matches Jellyfin standard (clean filename + correct metadata)
- CLEAN: track needs renaming/metadata fix (provide correct artist/title)
- SWAP_ARTIST_TITLE: artist and title are swapped; agent will swap and re-evaluate
- NEEDS_REVIEW: insufficient evidence, unclear artist, or unverified artist

SCHEMA (return EXACTLY this JSON array, no markdown, no prose):
{
  "tracks": [
    {
      "path": "...",
      "action": "KEEP|CLEAN|SWAP_ARTIST_TITLE|NEEDS_REVIEW",
      "artist": "Real Artist Name",
      "title": "Real Song Title",
      "track_number": 1,
      "confidence": "HIGH|MEDIUM|LOW",
      "reason": "Concise operational reason"
    },
    ...
  ],
  "batch_warnings": ["..."]
}

ABSOLUTE FORMAT RULES:
- Return exactly one JSON object with a "tracks" array and optional "batch_warnings" array
- Do not wrap in markdown code fences
- Do not include explanations, commentary, or prose
- Do not include trailing commas or comments
- "path" MUST match the input track path exactly
- "artist" and "title" MUST be filled with your identified real artist/title
- "track_number" should be the track number (from filename or metadata)
"""
            },
            {
                    "role": "user",
                    "content": json.dumps({
                        "folder": str(folder),
                        "riddim": riddim,
                        "year": year,
                        "tracks": compact_tracks,
                        "artist_cache": artist_context
                    }, ensure_ascii=False, separators=(",", ":"))
                }
        ]

        message = self.llm.chat(
            messages,
            tools=None,
        )

        content = message.get("content", "").strip()
        ai_log(f"PLANNER: raw response (first 500 chars): {content[:500]}")

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
            plan_data = json.loads(content)
        except json.JSONDecodeError:
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1 and end > start:
                json_str = content[start:end+1]
                try:
                    plan_data = json.loads(json_str)
                except json.JSONDecodeError as exc:
                    error_log(
                        f"Planner returned invalid JSON for riddim {riddim}: {exc}\n"
                        f"Content (first 500 chars): {content[:500]}"
                    )
                    plan_data = {"tracks": [], "batch_warnings": ["Planner returned invalid JSON"]}

        if not isinstance(plan_data, dict) or "tracks" not in plan_data:
            error_log(f"Planner response was not a valid plan for {riddim}")
            plan_data = {"tracks": [], "batch_warnings": ["Planner response was not a valid plan"]}

        decisions = plan_data.get("tracks", [])
        batch_warnings = plan_data.get("batch_warnings", [])

        # Defense-in-depth: always use the actual filesystem path.
        # The LLM must never provide file paths; they are owned by the agent.
        for idx, decision in enumerate(decisions):
            if idx < len(compact_tracks):
                decision["path"] = compact_tracks[idx].get("path")
                decision["filename"] = compact_tracks[idx].get("filename")
            else:
                decision["path"] = decision.get("path", "")
                decision["filename"] = Path(decision.get("path", "")).name

            decision.setdefault("action", "NEEDS_REVIEW")
            decision.setdefault("artist", "")
            decision.setdefault("title", "")
            decision.setdefault("track_number", 0)
            decision.setdefault("confidence", "LOW")
            decision.setdefault("reason", "No supporting evidence found.")
            decision["riddim"] = riddim
            decision["year"] = year

            # Defense: if LLM says KEEP but filename doesn't match the
            # global Jellyfin standard (Artist - Title.ext), force CLEAN.
            if decision.get("action") == "KEEP":
                fn = Path(decision.get("filename", ""))
                stem = fn.stem
                if re.match(r'^\d{1,2}(\s*[-.]\s*|\s+)', stem) or not re.match(r'^.+\s*-\s*.+$', stem):
                    decision["action"] = "CLEAN"
                    decision["reason"] = (
                        f"Filename does not match Jellyfin clean format "
                        f"Artist - Title: {fn.name}"
                    )
                    decision["confidence"] = "HIGH"

            if not decision.get("filename"):
                decision["filename"] = Path(decision.get("path", "")).name

            plan["tracks"].append(decision)

        plan["batch_warnings"] = batch_warnings
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

SYSTEM_PROMPT = """You are an autonomous music-library cleanup agent.

Your job is to clean TRACK FILENAMES and EMBEDDED METADATA for Jellyfin.

IMPORTANT:
- NEVER rename folders.
- NEVER delete audio files.
- NEVER invent artist or title information.
- If evidence is insufficient, use NEEDS_REVIEW.
- You have tools. Use them when useful.
- Inspect sibling tracks when the current track is ambiguous.
- Existing correct metadata should normally be preserved.
- You are allowed to make decisions, but actual modifications require human approval.

Jellyfin target:
Title = Song Title
Artist = Artist
Album = riddim folder name
Album Artist = Various Artists
Year = parent year folder
Track Number = NN (stored in MP3 metadata only)
Genre = Dancehall

Confidence policy:
HIGH: Multiple pieces of evidence agree.
MEDIUM: Strong local evidence but incomplete confirmation.
LOW: Weak, conflicting, or speculative evidence.

IMPORTANT: Artist/Title identification rules (use these FIRST before any other reasoning):
- The real ARTIST is the person/group who performed the song (e.g. "Beenie Man", "Vybz Kartel").
- The real TITLE is the song name (e.g. "Ramping Shop", "Summer Time").
- The RIDDIM is the instrumental/beat name (e.g. "Joy Ride", "Bogle"). The riddim is NEVER the artist.
- If a filename says "RiddimName @ Performer", the performer is the ARTIST and the riddim is the ALBUM, not the artist.
- If metadata says artist="RiddimName" but the filename shows a real performer name, the metadata artist is likely the riddim — SWAP the values.
- Use filename AND metadata together: if they conflict, the filename usually indicates the correct artist/title split.
- If you cannot determine which is artist vs title from the evidence, use NEEDS_REVIEW.

ACTION SCHEMA:
You must return exactly one action per track. The action field is mandatory.

Valid actions:
- KEEP: Track metadata is already correct; no changes needed.
- CLEAN: Metadata needs fixing; set new artist/title/track metadata.
- SWAP_ARTIST_TITLE: The metadata artist and title are swapped (common in riddim music where the riddim name is stored as artist). Swap them so artist=real performer, title=song name.
- NEEDS_REVIEW: Insufficient evidence to decide; flag for manual review.

For KEEP and CLEAN actions, you must also provide:
- artist: the correct artist name (string, non-empty)
- title: the correct song title (string, non-empty)
- track_number: the correct track number (integer, or null if unknown)
- reason: why you chose this action
- confidence: HIGH / MEDIUM / LOW

For SWAP_ARTIST_TITLE, you must also provide:
- artist: the REAL artist name (after swap)
- title: the REAL song title (after swap)
- track_number: the correct track number (integer, or null if unknown)
- reason: explain the swap (e.g. "Metadata artist looks like riddim name; filename shows real performer")
- confidence: HIGH / MEDIUM / LOW

Do not explain private chain-of-thought.
Instead provide concise operational reasoning:
- what you observed
- what evidence you are using
- why you are calling a tool
- what decision you reached

### STRICT RULES SECTION

Throughout this task, you will encounter filename patterns that have established conventions. When a filename matches one of the following patterns, you MUST adhere to the specified rules. These rules are mandatory constraints, not suggestions.

Rules are loaded from config/rules.yaml at startup. Each rule has:
- A regex pattern to match against the filename
- Mandatory metadata values that must be set (these override default reasoning)
- Forbidden metadata values that must NOT be set
- Reason text to include in your decision

YOU MUST check for rule matches BEFORE making any decision based on evidence. If a filename matches a rule, the mandatory fields are non-negotiable. If it doesn't match any rule, proceed with normal evidence-based reasoning.

CURRENT RULES:
1. leading_track_number_dash: Matches "NN - Artist - Title" format.
   - Mandatory: Title=exact filename title, Artist=exact filename artist, Album=riddim folder name, Album Artist="Various Artists", Track Number=extracted from filename, Genre="Dancehall", Year=valid (1950-2030) or auto
   - Forbidden: Do NOT set Compilation or Disc Number
   - Reason: "Filename contains a leading track number: {filename}"

2. artist_dash_title_pattern: Matches "Artist - Title" format (no leading track number).
   - Mandatory: Title=exact filename title, Artist=exact filename artist, Album=riddim folder name, Album Artist="Various Artists", Track Number=1, Genre="Dancehall", Year=valid (1950-2030) or auto
   - Forbidden: Do NOT set Compilation or Disc Number
   - Reason: "Filename matches Artist - Title pattern: {filename}"

If you are uncertain whether a rule applies, err on the side of checking. When in doubt, match the pattern case-insensitively.

YOU CANNOT OVERRIDE MANDATORY FIELDS FROM A MATCHED RULE. Even if your evidence suggests different values, the rule's mandatory values take precedence. You may only add optional metadata not covered by the rule, or choose NEEDS_REVIEW if the rule truly doesn't apply or cannot be overridden.

### ARTIST VERIFICATION CONTEXT

You will receive a list of cached artist verification results as "artist_cache". Each entry contains:
- artist_name: the name that was searched
- verified: whether the name was verified as a real artist
- source: "spotify", "youtube", "cache", or null
- spotify_id: Spotify artist ID if found
- youtube_channel_id: YouTube channel ID if found

Use this context to:
1. Confirm whether an artist name is a real performer or a riddim/instrumental name
2. Detect when metadata artist looks like a riddim name (contains "Riddim", "Diwali", "Bogle", "Dash", etc.) but the filename shows a real performer
3. Decide if a SWAP_ARTIST_TITLE is needed

If the artist is NOT verified, the agent will run a mandatory verification search after your decision. You do not need to search yourself — just identify the correct artist/title and flag for verification.

### INPUT FORMAT

You will receive the following context for each track:
- track_path: full file path
- filename: just the filename
- riddim: the riddim folder name
- year: the year folder
- metadata: embedded metadata (artist, title, album, etc.)
- filename_artist: artist parsed from filename (may be wrong)
- filename_title: title parsed from filename (may be wrong)
- filename_track_number: track number parsed from filename
- filename_source: which pattern matched the filename
- artist_cache: list of cached verification results for related artist names
- siblings: list of sibling tracks with their metadata (for context)

Use ALL available evidence to determine the correct artist and title. Do not rely solely on regex-parsed values from the filename — use the raw filename and metadata together to identify the real performer and song name."""


class RiddimAgent:

    def __init__(self, llm: LLM, memory: Memory):

        self.llm = llm
        self.memory = memory
        self.tools = ToolExecutor(memory)

        # Track number auto-assignment state
        self._track_num_counter = {}      # riddim -> next available number
        self._used_track_nums = {}        # riddim -> set of used numbers

    def _format_summary_value(self, value):
        if value is None or value == "":
            return "SKIPPED"

        if isinstance(value, list):
            value = value[0] if value else None

        text = str(value).strip()
        if text in {"", "UNKNOWN", "N/A", "NULL", "None"}:
            return "SKIPPED"
        return text

    def print_track_summary(self, path: Path, status: str = "UNKNOWN", reason: str | None = None, metadata: dict | None = None):
        """Print a concise, verified summary of the actual file state after processing."""
        safe_path = Path(path)
        actual_metadata = metadata or read_metadata(safe_path)

        if "error" in actual_metadata:
            title = "SKIPPED"
            artist = "SKIPPED"
            album = "SKIPPED"
            albumartist = "SKIPPED"
            track = "SKIPPED"
            year = "SKIPPED"
            genre = "SKIPPED"
            compilation = "SKIPPED"
            disc = "SKIPPED"
        else:
            title = self._format_summary_value(actual_metadata.get("title"))
            artist = self._format_summary_value(actual_metadata.get("artist"))
            album = self._format_summary_value(actual_metadata.get("album"))
            albumartist = self._format_summary_value(actual_metadata.get("albumartist"))
            track = self._format_summary_value(actual_metadata.get("tracknumber"))
            year = self._format_summary_value(actual_metadata.get("date"))
            genre = self._format_summary_value(actual_metadata.get("genre"))
            compilation = self._format_summary_value(actual_metadata.get("compilation"))
            disc = self._format_summary_value(actual_metadata.get("discnumber"))

        print()
        print(f"STATUS: {status.upper()}")
        print("FILE:")
        print(f"    {safe_path.name}")
        print("METADATA:")
        print(f"    Title        = {title}")
        print(f"    Artist       = {artist}")
        print(f"    Album        = {album}")
        print(f"    Album Artist = {albumartist}")
        print(f"    Track Number = {track}")
        print(f"    Year         = {year}")
        print(f"    Genre        = {genre}")
        print(f"    Compilation  = {compilation}")
        print(f"    Disc Number  = {disc}")
        if reason:
            print("REASON:")
            print(f"    {reason}")
        print()

        return {
            "status": status,
            "file": safe_path.name,
            "metadata": {
                "title": title,
                "artist": artist,
                "album": album,
                "albumartist": albumartist,
                "tracknumber": track,
                "year": year,
                "genre": genre,
                "compilation": compilation,
                "discnumber": disc,
            },
            "reason": reason,
        }

    def _check_strict_rules(self, path: Path, riddim: str, year: str | None) -> dict | None:
        """Check if the filename matches any strict rule.
        Returns a decision dict if a rule matches, or None if no rule applies.
        """
        stem = path.stem
        filename = path.name

        for rule in RULES:
            pattern = rule.get("match_pattern")
            if not pattern:
                continue

            try:
                match = re.match(pattern, stem)
            except re.error:
                continue

            if not match:
                continue

            # Rule matched - build the mandatory decision
            groups = match.groupdict()

            # Build metadata from mandatory fields
            mandatory = rule.get("mandatory_metadata", {})
            metadata = {}

            for key, value_template in mandatory.items():
                if isinstance(value_template, str):
                    # Resolve template variables
                    if value_template == "{folder_name}":
                        metadata[key] = riddim
                    elif value_template == "{artist}":
                        metadata[key] = groups.get("artist", "").strip()
                    elif value_template == "{title}":
                        metadata[key] = groups.get("title", "").strip()
                    elif value_template == "{track_number}":
                        try:
                            metadata[key] = int(groups.get("track_number", "1"))
                        except (ValueError, TypeError):
                            metadata[key] = 1
                    elif value_template == "auto":
                        metadata[key] = "auto"
                    else:
                        # Literal value
                        metadata[key] = value_template
                else:
                    metadata[key] = value_template

            # Apply year validation if year is "auto"
            if metadata.get("year") == "auto":
                year_validation = rule.get("year_validation", {})
                min_year = year_validation.get("min", 1950)
                max_year = year_validation.get("max", 2030)
                on_invalid = year_validation.get("on_invalid", "auto")

                if year is not None:
                    try:
                        year_int = int(year)
                        if min_year <= year_int <= max_year:
                            metadata["year"] = year
                        else:
                            # Year out of range
                            if on_invalid == "needs_review":
                                return None  # fall through to normal reasoning
                            else:
                                metadata["year"] = "auto"
                    except (ValueError, TypeError):
                        metadata["year"] = "auto"
                else:
                    metadata["year"] = "auto"

            # Build reason
            reason_template = rule.get("reason_template", "Filename matched rule: {rule_name}")
            reason = reason_template.format(
                filename=filename,
                rule_name=rule.get("name", "unknown")
            )

            decision = {
                "action": "CLEAN",
                "metadata": metadata,
                "reason": reason,
                "confidence": rule.get("confidence", "HIGH"),
                "rule_applied": rule.get("name", "unknown"),
                "forbidden": rule.get("forbidden_metadata", []),
            }

            return decision

        return None

    def process_track(
        self,
        path: Path,
        riddim: str,
        year: str | None
    ):
        try:
            fp = fingerprint(path)

            previous = self.memory.get_track(fp)

            if previous:
                if previous["status"] in {"completed", "approved", "compliant"}:
                    log(
                        f"Skipping previously processed track: "
                        f"{path.name}"
                    )
                    # Log skipped track
                    if processing_log is not None:
                        meta = read_metadata(path)
                        processing_log.log_track(
                            path=path,
                            before=meta,
                            after=meta,
                            status="SKIPPED",
                            reason="Previously processed (completed/approved/compliant)",
                            rule_applied="",
                            decision="SKIP",
                            track_number_before=meta.get("tracknumber"),
                            track_number_after=meta.get("tracknumber"),
                        )
                    return

            metadata = read_metadata(path)
            fn_data = self._parse_filename(path)

            # Check strict rules before fast paths or AI reasoning
            rule_decision = self._check_strict_rules(path, riddim, year)
            if rule_decision:
                # Merge rule metadata into decision
                decision = rule_decision
                action = "CLEAN"
            else:
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

                if re.match(r'^\d{1,2}(\s*[-.]\s*|\s+)', path.stem):
                    ai_log(
                        f"Fast-path CLEAN for leading-track filename: {path.name}"
                    )
                    decision = {
                        "action": "CLEAN",
                        "confidence": "HIGH",
                        "reason": f"Filename contains a leading track number: {path.stem}",
                    }
                    action = "CLEAN"
                    skip_llm = True
                else:
                    ai_log(
                        f"Investigating: {path.name}"
                    )
                    skip_llm = False

                if not skip_llm:
                    # Collect sibling tracks for context
                    siblings = []
                    try:
                        for sibling in path.parent.iterdir():
                            if sibling.is_file() and sibling != path:
                                sm = read_metadata(sibling)
                                if "error" not in sm:
                                    siblings.append({
                                        "filename": sibling.name,
                                        "artist": sm.get("artist", ""),
                                        "title": sm.get("title", ""),
                                    })
                    except Exception:
                        pass

                    # Build artist_cache from local DB
                    artist_cache = []
                    candidate_names = set()
                    candidate_names.add(fn_data["artist"])
                    candidate_names.add(metadata.get("artist", ""))
                    candidate_names.add(riddim)
                    for sib in siblings:
                        candidate_names.add(sib.get("artist", ""))
                        candidate_names.add(sib.get("title", ""))
                    for name in candidate_names:
                        if name and name.strip():
                            cached = self.memory.get_artist_cache(name.strip())
                            artist_cache.append({
                                "artist_name": name.strip(),
                                "verified": cached.get("verified", 0) if cached else 0,
                                "source": cached.get("source", "none") if cached else "none",
                                "spotify_id": cached.get("spotify_id") if cached else None,
                                "youtube_channel_id": cached.get("youtube_channel_id") if cached else None,
                            })

                    context = {
                        "track_path": str(path),
                        "filename": path.name,
                        "riddim": riddim,
                        "year": year,
                        "metadata": metadata,
                        "filename_artist": fn_data["artist"],
                        "filename_title": fn_data["title"],
                        "filename_track_number": fn_data["track_number"],
                        "filename_source": fn_data["source"],
                        "artist_cache": artist_cache,
                        "siblings": siblings,
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

                    # Mandatory artist verification AFTER LLM decision
                    # (as per user requirement: verification happens after LLM identifies)
                    verified_artist = None
                    if action in ("KEEP", "CLEAN", "SWAP_ARTIST_TITLE"):
                        candidate_artist = decision.get("artist", "") or fn_data["artist"] or metadata.get("artist") or ""
                        if candidate_artist:
                            verified_artist = self.verify_artist(candidate_artist, riddim)
                            if not verified_artist.get("verified"):
                                ai_log(
                                    f"Artist '{candidate_artist}' NOT verified for {path.name} "
                                    f"(source: {verified_artist.get('source', 'none')}). Forcing NEEDS_REVIEW."
                                )
                                decision["needs_review"] = True
                                decision["reason"] = (
                                    f"Artist '{candidate_artist}' could not be verified via Spotify/YouTube. "
                                    f"Manual review required."
                                )
                                action = "NEEDS_REVIEW"

            if action == "KEEP":
                # Use the LLM's identified artist/title as ground truth (verified by agent-side search)
                metadata_artist = decision.get("artist", "")
                metadata_title = decision.get("title", "")

                # Fallback: if LLM didn't provide values, use embedded metadata first, then filename-derived values.
                # Embedded metadata is the ground truth; filename values may have incorrect case.
                if not metadata_artist or not metadata_title:
                    metadata_artist = metadata.get("artist") or fn_data["artist"] or ""
                    metadata_title = metadata.get("title") or fn_data["title"] or ""

                # Use the same logic as CLEAN path for track number
                metadata_tracknumber = metadata.get("tracknumber")
                try:
                    if isinstance(metadata_tracknumber, list):
                        metadata_tracknumber = metadata_tracknumber[0] if metadata_tracknumber else None
                    if metadata_tracknumber is not None:
                        metadata_tracknumber = int(str(metadata_tracknumber).split("/")[0].strip())
                except Exception:
                    metadata_tracknumber = None

                # Prefer the parsed filename track number if present
                if fn_data["track_number"] is not None:
                    track_number = fn_data["track_number"]
                elif metadata_tracknumber is not None:
                    track_number = metadata_tracknumber
                else:
                    track_number = self._next_track_number(riddim)

                if track_number:
                    if track_number in self._used_track_nums[riddim]:
                        track_number = self._next_track_number(riddim)
                    else:
                        self._used_track_nums[riddim].add(track_number)
                        if track_number >= self._track_num_counter.get(riddim, 1):
                            self._track_num_counter[riddim] = track_number + 1

                # Build new metadata dict matching the strict-rule schema
                new_metadata = {
                    "title": metadata_title.strip(),
                    "artist": metadata_artist.strip(),
                    "album": riddim,
                    "albumartist": "Various Artists",
                    "date": year if year else "",
                    "tracknumber": str(track_number),
                    "genre": "Dancehall",
                }

                # Write the new metadata to the file
                write_metadata(path, new_metadata)

                # Read it back to confirm and display
                verified = read_metadata(path)
                self.print_track_summary(
                    path,
                    status="ALREADY_CORRECT",
                    reason=decision.get("reason", "Track already matches the target state."),
                    metadata=verified,
                )
                # Log unchanged track
                if processing_log is not None:
                    processing_log.log_track(
                        path=path,
                        before=metadata,
                        after=verified,
                        status="ALREADY_CORRECT",
                        reason=decision.get("reason", "Track already matches the target state."),
                        rule_applied="",
                        decision="KEEP",
                        track_number_before=metadata.get("tracknumber"),
                        track_number_after=verified.get("tracknumber"),
                    )
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
                verified = read_metadata(path)
                self.print_track_summary(
                    path,
                    status="NEEDS_REVIEW",
                    reason=decision.get("reason", "Insufficient evidence."),
                    metadata=verified,
                )
                # Log unchanged track needing review
                if processing_log is not None:
                    processing_log.log_track(
                        path=path,
                        before=metadata,
                        after=verified,
                        status="NEEDS_REVIEW",
                        reason=decision.get("reason", "Insufficient evidence."),
                        rule_applied="",
                        decision="NEEDS_REVIEW",
                        track_number_before=metadata.get("tracknumber"),
                        track_number_after=verified.get("tracknumber"),
                    )
                return

            if action == "SWAP_ARTIST_TITLE":
                # Swap artist and title as identified by LLM
                swapped_artist = decision.get("title", "")
                swapped_title = decision.get("artist", "")

                ai_log(
                    f"SWAP ARTIST/TITLE: '{decision.get('artist', '')}' -> artist, "
                    f"'{decision.get('title', '')}' -> title for {path.name}"
                )

                # Use swapped values
                metadata_artist = swapped_artist
                metadata_title = swapped_title

                # Fall back to metadata if LLM didn't provide both
                if not metadata_artist or not metadata_title:
                    # Use embedded metadata first (ground truth), then filename-derived values.
                    # Rule-decision metadata is NOT used because it's derived from the filename
                    # (which may have incorrect case), whereas embedded metadata is the ground truth.
                    metadata_artist = metadata_artist or metadata.get("artist") or fn_data["artist"] or ""
                    metadata_title = metadata_title or metadata.get("title") or fn_data["title"] or ""

                artist = metadata_artist.strip()
                title = metadata_title.strip()

                # Track number: same logic as CLEAN
                if decision.get("track_number") is not None:
                    track_number = decision.get("track_number")
                else:
                    metadata_tracknumber = metadata.get("tracknumber")
                    try:
                        if isinstance(metadata_tracknumber, list):
                            metadata_tracknumber = metadata_tracknumber[0] if metadata_tracknumber else None
                        if metadata_tracknumber is not None:
                            metadata_tracknumber = int(str(metadata_tracknumber).split("/")[0].strip())
                    except Exception:
                        metadata_tracknumber = None

                    if fn_data["track_number"] is not None:
                        track_number = fn_data["track_number"]
                    elif metadata_tracknumber is not None:
                        track_number = metadata_tracknumber
                    else:
                        track_number = self._next_track_number(riddim)

                if riddim not in self._used_track_nums:
                    self._used_track_nums[riddim] = set()
                    self._track_num_counter[riddim] = 1

                if track_number:
                    if track_number in self._used_track_nums[riddim]:
                        track_number = self._next_track_number(riddim)
                    else:
                        self._used_track_nums[riddim].add(track_number)
                        if track_number >= self._track_num_counter[riddim]:
                            self._track_num_counter[riddim] = track_number + 1
                else:
                    track_number = self._next_track_number(riddim)

                # Verify we have required data before proceeding
                if not artist or not title:
                    self.memory.upsert_track({
                        "fingerprint": fp,
                        "path": str(path),
                        "riddim": riddim,
                        "year": year,
                        "status": "needs_review",
                        "last_decision": "NEEDS_REVIEW",
                        "confidence": "LOW",
                        "reason": "Cannot determine artist/title after swap",
                    })
                    return

                self.tools.pending_proposals.append({
                    "path": str(path),
                    "artist": artist,
                    "title": title,
                    "track_number": int(track_number),
                    "confidence": decision.get("confidence", "HIGH"),
                    "reason": f"Swapped artist/title. Original reason: {decision.get('reason', 'Swap detected by LLM')}",
                    "rule_applied": decision.get("rule_applied"),
                    "rule_metadata": decision.get("metadata"),
                })
                self.memory.upsert_track({
                    "fingerprint": fp,
                    "path": str(path),
                    "riddim": riddim,
                    "year": year,
                    "status": "pending_review",
                    "last_decision": "CLEAN",
                    "confidence": decision.get("confidence", "HIGH"),
                    "reason": f"Swapped artist/title: {decision.get('reason', 'Swap detected')}",
                    "artist": artist,
                    "title": title,
                    "album": riddim,
                    "rule_applied": decision.get("rule_applied"),
                })
                log(f"Swapped artist/title for: {path.name}")
                log(f"  Old: artist='{metadata.get('artist')}', title='{metadata.get('title')}'")
                log(f"  New: artist='{artist}', title='{title}'")

            if action == "CLEAN":
                # Use the LLM's identified artist/title as ground truth (verified by agent-side search)
                metadata_artist = decision.get("artist", "")
                metadata_title = decision.get("title", "")

                # Fallback: if LLM didn't provide values, use current file metadata, then filename-derived values.
                # Rule-decision metadata is NOT used for artist/title because it's derived from the filename
                # (which may have incorrect case), whereas embedded metadata is the ground truth.
                if not metadata_artist or not metadata_title:
                    metadata_artist = metadata_artist or metadata.get("artist") or fn_data["artist"] or ""
                    metadata_title = metadata_title or metadata.get("title") or fn_data["title"] or ""
                    artist = metadata_artist or (fn_data["artist"] or "").strip()
                    title = metadata_title or (fn_data["title"] or "").strip()
                else:
                    artist = metadata_artist.strip()
                    title = metadata_title.strip()

                # Use the LLM's track number if provided, otherwise existing logic
                if decision.get("track_number") is not None:
                    track_number = decision.get("track_number")
                else:
                    metadata_tracknumber = metadata.get("tracknumber")
                    if isinstance(metadata_tracknumber, list):
                        metadata_tracknumber = metadata_tracknumber[0] if metadata_tracknumber else None

                    try:
                        if metadata_tracknumber is not None:
                            metadata_tracknumber = int(str(metadata_tracknumber).split("/")[0].strip())
                    except Exception:
                        metadata_tracknumber = None

                    # Prefer the parsed filename track number if present, otherwise fall back
                    # to embedded metadata track number, and finally assign the next available.
                    if fn_data["track_number"] is not None:
                        track_number = fn_data["track_number"]
                    elif metadata_tracknumber is not None:
                        track_number = metadata_tracknumber
                    else:
                        track_number = self._next_track_number(riddim)

                # Initialize riddim state if needed
                if riddim not in self._used_track_nums:
                    self._used_track_nums[riddim] = set()
                    self._track_num_counter[riddim] = 1

                # Handle track number: use provided or auto-assign, check for conflicts
                if track_number:
                    if track_number in self._used_track_nums[riddim]:
                        track_number = self._next_track_number(riddim)
                    else:
                        self._used_track_nums[riddim].add(track_number)
                        if track_number >= self._track_num_counter[riddim]:
                            self._track_num_counter[riddim] = track_number + 1
                else:
                    track_number = self._next_track_number(riddim)

                # Verify we have required data before proceeding
                if not artist or not title:
                    self.memory.upsert_track({
                        "fingerprint": fp,
                        "path": str(path),
                        "riddim": riddim,
                        "year": year,
                        "status": "needs_review",
                        "last_decision": "NEEDS_REVIEW",
                        "confidence": "LOW",
                        "reason": "Cannot determine artist/title from filename or metadata",
                    })
                    return

                self.tools.pending_proposals.append({
                    "path": str(path),
                    "artist": artist,
                    "title": title,
                    "track_number": int(track_number),
                    "confidence": decision.get("confidence", "HIGH"),
                    "reason": decision.get("reason", "Track needs cleaning."),
                    "rule_applied": decision.get("rule_applied"),
                    "rule_metadata": decision.get("metadata"),
                })
                self.memory.upsert_track({
                    "fingerprint": fp,
                    "path": str(path),
                    "riddim": riddim,
                    "year": year,
                    "status": "pending_review",
                    "last_decision": "CLEAN",
                    "confidence": decision.get("confidence", "HIGH"),
                    "reason": decision.get("reason", "Track needs cleaning."),
                    "artist": artist,
                    "title": title,
                    "album": riddim,
                    "rule_applied": decision.get("rule_applied"),
                })
                verified = read_metadata(path)
                self.print_track_summary(
                    path,
                    status="PENDING_REVIEW",
                    reason=decision.get("reason", "Track needs cleaning."),
                    metadata=verified,
                )
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

        except Exception as e:
            error_log_detailed(
                f"Error processing {path.name} in {riddim}: {e}",
                exc_info=sys.exc_info()
            )
            if processing_log:
                processing_log.log_error(
                    f"Error processing {path.name} in {riddim}: {e}",
                    exc_info=sys.exc_info()
                )
            try:
                fp = locals().get("fp", str(path))
                self.memory.upsert_track({
                    "fingerprint": fp,
                    "path": str(path),
                    "riddim": riddim,
                    "year": year,
                    "status": "error",
                    "last_decision": "ERROR",
                    "confidence": "LOW",
                    "reason": f"Runtime error: {e}",
                })
            except Exception:
                pass

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
                if action in ("KEEP", "CLEAN", "NEEDS_REVIEW", "SWAP_ARTIST_TITLE"):
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
                "update.*to",
                "swap.*artist.*title",
                "artist.*title.*swap",
                "artist and title are swap",
            ]
            
            compliance_score = sum(1 for indicator in compliance_indicators if re.search(indicator, content_lower))
            non_compliance_score = sum(1 for indicator in non_compliance_indicators if re.search(indicator, content_lower))
            
            # Check for explicit swap intent in prose
            swap_indicators = [
                "swap",
                "swapped",
                "swap.*artist.*title",
                "artist.*is.*actually.*title",
                "title.*is.*actually.*artist",
                "metadata.*artist.*looks.*like.*riddim",
            ]
            swap_score = sum(1 for indicator in swap_indicators if re.search(indicator, content_lower))
            
            if swap_score > 0 and non_compliance_score > 0:
                action = "SWAP_ARTIST_TITLE"
            elif compliance_score > non_compliance_score and compliance_score > 0:
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

        # Only extract action, confidence, and reason from LLM response.
        # Artist/title/track_number come from filename/metadata ground truth.
        reason = extract("reason")
        if reason:
            decision["reason"] = reason

        # Remove any path/filename the LLM might have hallucinated.
        # The agent owns all file paths.
        decision.pop("path", None)
        decision.pop("filename", None)

        return decision

    def _parse_filename(self, path: Path) -> dict:
        """Extract artist, title, track_number from filename using regex.
        Returns dict with keys: track_number, artist, title, source.
        """
        stem = path.stem
        
        # Dash format: 06 - Major Damage - Tell Me What you Like
        m = re.match(r'^(\d{1,2})\s*-\s*(.+?)\s*-\s*(.+)$', stem)
        if m:
            return {"track_number": int(m.group(1)), "artist": m.group(2).strip(), 
                    "title": m.group(3).strip(), "source": "filename_dash"}
        
# Dot format: 01. Sizzla - Thanks & Praise
        m = re.match(r'^(\d{1,2})\.\s*(.+?)\s*-\s*(.+)$', stem)
        if m:
            return {"track_number": int(m.group(1)), "artist": m.group(2).strip(), 
                    "title": m.group(3).strip(), "source": "filename_dot"}

        # Space format: NN Artist - Title (missing dash after track number)
        m = re.match(r'^(\d{1,2})\s+(.+?)\s*-\s*(.+)$', stem)
        if m:
            return {"track_number": int(m.group(1)), "artist": m.group(2).strip(),
                    "title": m.group(3).strip(), "source": "filename_space"}

        # No number: Brian & Tony Gold - Champion OR Artist - Title
        # Also handle Artist # Title and Artist @ Title formats common in riddim packs
        m = re.match(r'^(.+?)\s*[-.]\s*(.+)$', stem)
        if m:
            artist = m.group(1).strip()
            title = m.group(2).strip()
            if artist and title:
                return {"track_number": None, "artist": artist, "title": title, "source": "filename_no_num"}
        
        # Handle Artist # Title pattern
        m = re.match(r'^(.+?)\s*#\s*(.+)$', stem)
        if m:
            artist = m.group(1).strip()
            title = m.group(2).strip()
            if artist and title:
                return {"track_number": None, "artist": artist, "title": title, "source": "filename_hash"}

        # Handle Artist @ Title pattern
        m = re.match(r'^(.+?)\s*@\s*(.+)$', stem)
        if m:
            artist = m.group(1).strip()
            title = m.group(2).strip()
            if artist and title:
                return {"track_number": None, "artist": artist, "title": title, "source": "filename_at"}

        return {"track_number": None, "artist": "", "title": "", "source": "none"}

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

    def _find_pending_proposal(self, path: Path):
        target = str(path)
        for proposal in self.tools.pending_proposals:
            if proposal.get("path") == target:
                return proposal
        return None

    def verify_single_track(self, path: Path) -> list[dict]:
        stem = path.stem

        if re.match(r'^\d{1,2}(\s*[-.]\s*|\s+)', stem):
            return [{
                "path": str(path),
                "filename": path.name,
                "reason": f"Filename contains a leading track number: {path.stem}"
            }]

        # Try the same patterns as _parse_filename in order of preference
        match = None
        artist = ""
        title = ""

        # Dash format: Artist - Title (preferred)
        match = re.match(r'^(.+?)\s*[-.]\s*(.+)$', stem)
        if match:
            artist = match.group(1).strip()
            title = match.group(2).strip()
        
        # Hash format: Artist # Title
        if not (artist and title):
            match = re.match(r'^(.+?)\s*#\s*(.+)$', stem)
            if match:
                artist = match.group(1).strip()
                title = match.group(2).strip()
        
        # At format: Artist @ Title
        if not (artist and title):
            match = re.match(r'^(.+?)\s*@\s*(.+)$', stem)
            if match:
                artist = match.group(1).strip()
                title = match.group(2).strip()

        if not match or not artist or not title:
            return [{
                "path": str(path),
                "filename": path.name,
                "reason": f"Does not match supported Artist - Title patterns: {path.stem}"
            }]

        # Check if track_number metadata is present and valid (required by strict rules)
        # Use the same logic as in investigate_metadata and process_track
        metadata = read_metadata(path)
        tracknumber = metadata.get("tracknumber")
        try:
            if isinstance(tracknumber, list):
                tracknumber = tracknumber[0] if tracknumber else None
            if tracknumber is not None:
                tracknumber = int(str(tracknumber).split("/")[0].strip())
        except Exception:
            tracknumber = None
        if tracknumber is None or tracknumber == 0:
            return [{
                "path": str(path),
                "filename": path.name,
                "reason": f"Missing or invalid track number metadata for: {path.stem}"
            }]

        return []

    def _apply_single_proposal(self, proposal: dict):
        path = Path(proposal["path"])

        try:
            original_metadata = read_metadata(path)
            new_name = make_filename(
                proposal["track_number"],
                proposal["artist"],
                proposal["title"],
                path.suffix
            )
            new_path = path.parent / new_name

            # DEBUG
            print(f"DEBUG: path.name={path.name!r}, new_name={new_name!r}, equal={path.name == new_name}")

            if path.name == new_name:
                failures = self.verify_single_track(path)
                if failures:
                    raise RuntimeError(
                        f"No-op rename generated for non-compliant filename: {path.name}"
                    )

            if new_path.exists() and new_path != path:
                raise FileExistsError(
                    f"Target already exists: {new_path}"
                )

            rule_metadata = proposal.get("rule_metadata") or {}
            year_value = rule_metadata.get("year") if rule_metadata.get("year") not in (None, "auto") else get_riddim_context(path)["year"]
            new_metadata = {
                "title": proposal["title"],
                "artist": proposal["artist"],
                "album": path.parent.name,
                "albumartist": "Various Artists",
                "date": year_value,
                "tracknumber": str(proposal["track_number"]),
                "genre": "Dancehall",
            }
            if proposal.get("rule_applied"):
                forbidden = {
                    "compilation",
                    "discnumber",
                }
                for field in forbidden:
                    new_metadata[field] = None

            write_metadata(path, new_metadata)

            if path.name != new_name:
                path.rename(new_path)

            rule_applied = proposal.get("rule_applied")
            upsert_data = {
                "fingerprint": fingerprint(new_path),
                "path": str(new_path),
                "riddim": path.parent.name,
                "year": new_metadata["date"],
                "status": "completed",
                "last_decision": "APPLIED",
                "confidence": proposal["confidence"],
                "artist": proposal["artist"],
                "title": proposal["title"],
                "album": path.parent.name,
                "original_path": str(path),
                "new_path": str(new_path),
                "reason": proposal.get("reason", "Track cleaned."),
            }
            if rule_applied:
                upsert_data["rule_applied"] = rule_applied
            self.memory.upsert_track(upsert_data)

            final_metadata = read_metadata(new_path)
            self.print_track_summary(
                new_path,
                status="UPDATED",
                reason=proposal.get("reason", "Track cleaned."),
                metadata=final_metadata,
            )

            # Log detailed track processing
            if processing_log is not None:
                # Determine track numbers before and after
                try:
                    tb = original_metadata.get("tracknumber")
                    if isinstance(tb, list):
                        tb = tb[0] if tb else None
                    if tb is not None:
                        tb = int(str(tb).split("/")[0].strip())
                except Exception:
                    tb = None
                try:
                    ta = final_metadata.get("tracknumber")
                    if isinstance(ta, list):
                        ta = ta[0] if ta else None
                    if ta is not None:
                        ta = int(str(ta).split("/")[0].strip())
                except Exception:
                    ta = None
                processing_log.log_track(
                    path=new_path,
                    before=original_metadata,
                    after=final_metadata,
                    status="UPDATED",
                    reason=proposal.get("reason", "Track cleaned."),
                    rule_applied=proposal.get("rule_applied", ""),
                    decision="CLEAN",
                    track_number_before=tb,
                    track_number_after=ta,
                )

            log(
                f"APPLIED: {path.name} -> {new_path.name}",
                "EXECUTE"
            )
            return {"success": [str(new_path)], "failed": [], "errors": []}

        except Exception as exc:
            error_log_detailed(f"Failed to modify {path}: {exc}", exc_info=sys.exc_info())
            if processing_log is not None:
                processing_log.log_error(f"Failed to modify {path}: {exc}", exc_info=sys.exc_info())
            return {"success": [], "failed": [str(path)], "errors": [f"{path.name}: {exc}"]}

    def fix_track_until_verified(self, path: Path, riddim: str, year: str | None):
        current_path = Path(path)

        for attempt in range(5):
            failures = self.verify_single_track(current_path)
            if not failures:
                ai_log(f"Track already verified: {current_path.name}")
                return True

            ai_log(
                f"Fix-Verify-Fix loop for {current_path.name} (attempt {attempt + 1})"
            )

            self.process_track(
                path=current_path,
                riddim=riddim,
                year=year
            )

            proposal = self._find_pending_proposal(current_path)
            if proposal is None:
                ai_log(
                    f"No proposal generated for {current_path.name}; leaving track unresolved."
                )
                return False

            result = self._apply_single_proposal(proposal)
            self.tools.pending_proposals = [
                p for p in self.tools.pending_proposals
                if p is not proposal
            ]

            if result["failed"]:
                self.memory.upsert_track({
                    "fingerprint": fingerprint(current_path),
                    "path": str(current_path),
                    "riddim": riddim,
                    "year": year,
                    "status": "needs_review",
                    "last_decision": "FIX_FAILED",
                    "confidence": "LOW",
                    "reason": "; ".join(result["errors"]),
                })
                return False

            if result["success"]:
                current_path = Path(result["success"][0])
                failures = self.verify_single_track(current_path)
                if not failures:
                    ai_log(
                        f"Track fixed and verified: {current_path.name}"
                    )
                    final_metadata = read_metadata(current_path)
                    self.print_track_summary(
                        current_path,
                        status="UPDATED",
                        reason="Track fixed and verified.",
                        metadata=final_metadata,
                    )
                    return True

        error_log(
            f"Track did not become compliant after fix attempts: {current_path.name}"
        )
        return False

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
            return {"success": [], "failed": [], "errors": []}

        if not AUTO_APPLY:
            answer = input(
                "\nApply these changes? [y/N]: "
            ).strip().lower()

            if answer != "y":
                log("Changes rejected by user.")
                return {"success": [], "failed": [], "errors": ["User rejected"]}

        if AUTO_APPLY:
            log("AUTO_APPLY enabled - applying without confirmation")

        batch_id = hashlib.sha256(
            str(time.time()).encode()
        ).hexdigest()[:16]

        log(f"Applying batch {batch_id}...")

        success = []
        failed = []
        errors = []

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
                    "date": (
                        (p.get("rule_metadata") or {}).get("year")
                        if (p.get("rule_metadata") or {}).get("year") not in (None, "auto")
                        else get_riddim_context(path)["year"]
                    ),
                    "tracknumber": str(p["track_number"]),
                    "genre": "Dancehall",
                }

                # Enforce forbidden fields from strict rules
                if p.get("rule_applied"):
                    forbidden = {"compilation", "discnumber"}
                    for field in forbidden:
                        new_metadata[field] = None

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

                final_metadata = read_metadata(new_path)

                # Log detailed track processing
                if processing_log is not None:
                    # Determine track numbers before and after
                    try:
                        tb = original_metadata.get("tracknumber")
                        if isinstance(tb, list):
                            tb = tb[0] if tb else None
                        if tb is not None:
                            tb = int(str(tb).split("/")[0].strip())
                    except Exception:
                        tb = None
                    try:
                        ta = final_metadata.get("tracknumber")
                        if isinstance(ta, list):
                            ta = ta[0] if ta else None
                        if ta is not None:
                            ta = int(str(ta).split("/")[0].strip())
                    except Exception:
                        ta = None
                    processing_log.log_track(
                        path=new_path,
                        before=original_metadata,
                        after=final_metadata,
                        status="UPDATED",
                        reason=p.get("reason", "Track cleaned."),
                        rule_applied=p.get("rule_applied", ""),
                        decision="CLEAN",
                        track_number_before=tb,
                        track_number_after=ta,
                    )

                log(
                    f"APPLIED: {path.name} -> {new_path.name}",
                    "EXECUTE"
                )
                success.append(str(new_path))

            except Exception as exc:
                error_log_detailed(f"Failed to modify {path}: {exc}", exc_info=sys.exc_info())
                if processing_log is not None:
                    processing_log.log_error(f"Failed to modify {path}: {exc}", exc_info=sys.exc_info())
                failed.append(str(path))
                errors.append(f"{path.name}: {exc}")

        self.tools.pending_proposals.clear()

        log(
            f"Batch {batch_id} finished. "
            f"Success: {len(success)}, Failed: {len(failed)}"
        )

        return {"success": success, "failed": failed, "errors": errors}

    def verify_riddim_folder(self, riddim_folder: Path) -> list[dict]:

        failures = []

        for f in sorted(riddim_folder.glob("*.mp3")):
            stem = f.stem

            if re.match(r'^\d{1,2}(\s*[-.]\s*|\s+)', stem):
                failures.append({
                    "path": str(f),
                    "filename": f.name,
                    "reason": f"Filename contains a leading track number: {stem}"
                })
                continue

            if not re.match(r'^.+\s*-\s*.+$', stem):
                failures.append({
                    "path": str(f),
                    "filename": f.name,
                    "reason": f"Does not match Artist - Title: {stem}"
                })

        return failures

    def reprocess_failed_tracks(
        self,
        failures: list[dict],
        riddim: str,
        year: str | None
    ):

        for fail in failures:
            path = Path(fail["path"])
            fp = fingerprint(path)

            if path.exists():
                self.memory.upsert_track({
                    "fingerprint": fp,
                    "path": str(path),
                    "riddim": riddim,
                    "year": year,
                    "status": "needs_review",
                    "last_decision": "REPROCESS",
                    "confidence": "LOW",
                    "reason": f"Re-processing after verification failure: {fail['reason']}",
                })

                log(
                    f"Re-processing: {path.name} - {fail['reason']}"
                )

                self.process_track(
                    path=path,
                    riddim=riddim,
                    year=year
                )


# ============================================================
# SCAN
# ============================================================

def scan_riddims(root: Path, memory: Memory | None = None):

    audio_extensions = {
        ".mp3",
        ".flac",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".wav",
    }

    def folder_has_audio(folder: Path) -> bool:
        try:
            for p in folder.iterdir():
                if p.is_file() and p.suffix.lower() in audio_extensions:
                    return True
        except OSError:
            return False
        return False

    def nearest_year(folder: Path):
        for parent in [folder, *folder.parents]:
            if parent.name.isdigit() and len(parent.name) == 4:
                return parent.name
        return None

    # Resume: skip folders already completed (from previous interrupted run)
    completed_paths = set()
    if memory is not None:
        try:
            rows = memory.conn.execute(
                "SELECT folder_path FROM folder_status WHERE status = 'completed'"
            ).fetchall()
            completed_paths = {row[0] for row in rows}
        except Exception:
            completed_paths = set()

    seen = set()

    for folder in sorted(root.rglob("*")):
        if not folder.is_dir():
            continue

        # Resume: skip folders already completed
        if str(folder.resolve()) in completed_paths:
            continue

        if not folder_has_audio(folder):
            continue

        norm = str(folder.resolve())
        if norm in seen:
            continue
        seen.add(norm)

        yield folder, nearest_year(folder)
        time.sleep(SCAN_DELAY_SECONDS)


# ============================================================
# MAIN
# ============================================================

def main():

    log("RIDDIM JELLYFIN AI AGENT")
    log("=" * 50)

    target_riddim = None
    if len(sys.argv) > 1:
        target_riddim = Path(sys.argv[1]).resolve()
        if not target_riddim.exists():
            raise FileNotFoundError(
                f"Target folder doesn't exist:\n{target_riddim}"
            )
        if not target_riddim.is_dir():
            raise NotADirectoryError(
                f"Target path is not a directory:\n{target_riddim}"
            )

    if not ROOT_FOLDER.exists():
        raise FileNotFoundError(
            f"Root folder doesn't exist:\n{ROOT_FOLDER}"
        )

    if target_riddim is not None:
        log(f"Target override: {target_riddim}")

    log(f"Root: {ROOT_FOLDER}")
    log(f"Model: {MODEL_PATH}")
    log(f"Memory: {STATE_DB}")

    # Initialize processing log
    global processing_log
    processing_log = ProcessingLog()
    log(f"Log file: {processing_log.log_path}")

    memory = Memory(STATE_DB)

    if target_riddim is not None:
        year = None
        if target_riddim.parent.name.isdigit() and len(target_riddim.parent.name) == 4:
            year = target_riddim.parent.name
        memory.upsert_folder_status(target_riddim, target_riddim.name, year)
        target_folders = [(target_riddim, year)]
    else:
        target_folders = []
        scan_iter = scan_riddims(ROOT_FOLDER, memory=memory)
        with tqdm(
            scan_iter,
            desc="Scanning riddims",
            unit="folder",
            leave=True,
        ) as pbar:
            for folder, y in pbar:
                pbar.set_postfix_str(folder.name)
                memory.upsert_folder_status(folder, folder.name, y)
                target_folders.append((folder, y))

    # Filter out already-completed folders so the global progress bar
    # reflects only the work remaining on this run.
    pending_folders = []
    for folder, y in target_folders:
        folder_status = memory.get_folder_progress(folder.name)
        if folder_status and folder_status.get("status") == "completed":
            log(f"Skipping already completed folder: {folder.name}")
            continue
        memory.upsert_folder_status(folder, folder.name, y)
        pending_folders.append((folder, y))

    log(f"Resuming: {len(pending_folders)} folder(s) remaining to process.")

    if target_riddim is not None:
        target_name = target_riddim.name
        target_prefix = str(target_riddim)

        memory.conn.execute(
            "DELETE FROM operations WHERE original_path LIKE ? OR new_path LIKE ?",
            (f"{target_prefix}%", f"{target_prefix}%")
        )
        memory.conn.execute(
            "DELETE FROM reviews WHERE path LIKE ?",
            (f"{target_prefix}%",)
        )
        memory.conn.execute(
            "DELETE FROM tracks WHERE riddim = ? OR path LIKE ? OR new_path LIKE ? OR original_path LIKE ?",
            (target_name, f"{target_prefix}%", f"{target_prefix}%", f"{target_prefix}%")
        )
        memory.conn.commit()

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

        # Global progress bar across all remaining folders.
        with tqdm(
            pending_folders,
            desc="Processing riddims",
            unit="folder",
            leave=True,
        ) as global_pbar:
            for riddim_folder, year in global_pbar:
                global_pbar.set_postfix_str(riddim_folder.name)
                # Skip entirely completed folders (defensive check).
                folder_status = memory.get_folder_progress(riddim_folder.name)
                if folder_status and folder_status.get("status") == "completed":
                    log(f"Skipping already completed folder: {riddim_folder.name}")
                    global_pbar.update(1)
                    continue

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

                memory.update_folder_track_count(riddim_folder, len(tracks))

                log(
                    f"{len(tracks)} tracks found."
                )

                # Seed track number counter from existing filenames
                # to avoid collisions with already-processed files
                for t in tracks:
                    m = re.match(r'^(\d+)\s*-\s', t["filename"])
                    if m:
                        num = int(m.group(1))
                        existing = agent._track_num_counter.get(
                            riddim_folder.name, 1
                        )
                        if num >= existing:
                            agent._track_num_counter[riddim_folder.name] = num + 1

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
                        memory.update_folder_progress(riddim_folder, riddim_folder.name, needs_review=True)
                        continue

                    if decision["action"] == "KEEP":
                        log(
                            f"Processing already-correct track: {path.name}"
                        )
                        memory.update_folder_progress(riddim_folder, riddim_folder.name, completed=True)
                        continue

                    if decision["action"] == "NEEDS_REVIEW":
                        log(
                            f"Processing review-needed track: {path.name}"
                        )
                        memory.update_folder_progress(riddim_folder, riddim_folder.name, needs_review=True)
                        continue

                    # Use planner's decision directly — skip redundant LLM call
                    if decision.get("artist") and decision.get("title"):
                        agent.tools.pending_proposals.append({
                            "path": str(path),
                            "artist": decision["artist"],
                            "title": decision["title"],
                            "track_number": decision.get("track_number") or 1,
                            "confidence": decision.get("confidence", "HIGH"),
                            "reason": decision.get("reason", ""),
                            "planner_decision": decision,
                        })
                        log(
                            f"Planner decision used for: {path.name} [{decision['action']}]"
                        )
                    else:
                        agent.fix_track_until_verified(
                            path=path,
                            riddim=riddim_folder.name,
                            year=year
                        )
                        log(
                            f"Track processed via LLM: {path.name}"
                        )
                    memory.update_folder_progress(riddim_folder, riddim_folder.name, completed=True)

                # --------------------------------------------------------
                # FIX-VERIFY-FIX LOOP
                # Apply, verify, reprocess failures, repeat until clean
                # --------------------------------------------------------

                if agent.tools.pending_proposals:
                    iteration = 0
                    while agent.tools.pending_proposals:
                        iteration += 1
                        log(
                            f"Fix-verify iteration {iteration}..."
                        )

                        try:
                            result = agent.apply_proposals()
                        except Exception as exc:
                            error_log_detailed(
                                f"apply_proposals failed for {riddim_folder.name}: {exc}",
                                exc_info=sys.exc_info()
                            )
                            if processing_log:
                                processing_log.log_error(
                                    f"apply_proposals failed for {riddim_folder.name}: {exc}",
                                    exc_info=sys.exc_info()
                                )
                            break

                        # Verify the riddim folder
                        failures = agent.verify_riddim_folder(
                            riddim_folder
                        )

                        if not failures:
                            log(
                                f"All tracks verified compliant "
                                f"after {iteration} iteration(s)."
                            )
                            break

                        log(
                            f"Iteration {iteration}: "
                            f"{len(failures)} track(s) need fixing."
                        )

                        agent.reprocess_failed_tracks(
                            failures,
                            riddim_folder.name,
                            year
                        )

                        if iteration >= 3:
                            error_log(
                                f"Max iterations reached; "
                                f"{len(failures)} track(s) still non-compliant."
                            )
                            break

                # --------------------------------------------------------
                # FOLDER STATUS UPDATE
                # --------------------------------------------------------
                folder_status = memory.get_folder_progress(riddim_folder.name)
                if folder_status and folder_status.get("status") != "completed":
                    unresolved = agent.memory.get_unresolved_tracks(riddim_folder.name)
                    if not unresolved:
                        memory.mark_folder_complete(riddim_folder, riddim_folder.name)
                        log(f"Folder marked complete: {riddim_folder.name}")
                    else:
                        memory.update_folder_progress(riddim_folder, riddim_folder.name, needs_review=True)
                        log(f"Folder remains in progress: {riddim_folder.name} ({len(unresolved)} unresolved)")

                # --------------------------------------------------------
                # UNRESOLVED TRACKS REPORT
                # --------------------------------------------------------

                unresolved = [
                    t for t in agent.memory.get_unresolved_tracks(
                        riddim_folder.name
                    )
                ]
                if unresolved:
                    log(
                        f"{len(unresolved)} track(s) remain unresolved "
                        f"in {riddim_folder.name}:"
                    )
                    for t in unresolved:
                        log(
                            f"  - {t.get('filename', t.get('path'))}: "
                            f"{t.get('reason', 'no reason')}"
                        )

        log("")
        log("=" * 70)
        log("COLLECTION PROCESSING COMPLETE")
        log("=" * 70)

    except KeyboardInterrupt:

        log("Interrupted by user.")

    except Exception as exc:
        error_log_detailed(str(exc), exc_info=sys.exc_info())

    finally:

        memory.close()
        server.stop()


if __name__ == "__main__":
    main()