"""
server.py — SIA Main Server
FastAPI + WebSocket voice pipeline.
Claude Haiku → ElevenLabs TTS (or macOS say fallback) → Browser
"""

import asyncio
import base64
import json
import logging
import os
import re
import ssl
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

import anthropic
import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Load env ─────────────────────────────────────────────────────────────────
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "hpp4J3VqNfWAUOO0d1Us")  # Sarah — Mature, Reassuring, Confident
USER_NAME = os.getenv("USER_NAME", "")

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("SIA")

# ── Anthropic clients ────────────────────────────────────────────────────────
if not ANTHROPIC_API_KEY:
    log.warning("ANTHROPIC_API_KEY not set — AI responses will fail.")

ai = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None
ai_async = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if ANTHROPIC_API_KEY else None

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="SIA", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Session state ─────────────────────────────────────────────────────────────
sessions: dict[str, dict] = {}

# Queue for rich results to send to frontend panel
_pending_panel_data: list[dict] = []


def get_session(session_id: str) -> dict:
    if session_id not in sessions:
        sessions[session_id] = {
            "id": session_id,
            "history": [],
            "created_at": time.time(),
        }
    return sessions[session_id]


# ── System prompt ────────────────────────────────────────────────────────────
def build_system_prompt() -> str:
    from datetime import datetime
    now = datetime.now()
    hour = now.hour
    if hour < 12:
        time_of_day = "morning"
    elif hour < 17:
        time_of_day = "afternoon"
    else:
        time_of_day = "evening"

    day_str = now.strftime("%A, %d %B %Y")
    time_str = now.strftime("%I:%M %p")
    name_line = f"You are speaking with {USER_NAME}. Address them by name occasionally." if USER_NAME else ""

    return f"""You are SIA — Structured Intent Agent.
You are a voice-first AI assistant running on Windows 11.
Your personality: calm, precise, witty, warm, and impeccably helpful.
You are NOT a British butler — you are a sharp, modern AI assistant with a natural conversational tone.

{name_line}

CURRENT TIME CONTEXT:
- It is {time_of_day} — {time_str} on {day_str}
- Use this for greetings and time-aware responses. Never say "good afternoon" in the morning.

WINDOWS ENVIRONMENT:
- You run on Windows 11. Use Windows terminology (File Explorer not Finder, PowerShell not Terminal, taskbar not dock).
- Apps open via Windows start menu or taskbar.
- File paths use backslashes e.g. C:\\Users\\Kaavya\\Desktop
- Use Windows keyboard shortcuts (Win+D for desktop, Win+E for Explorer, etc.)
- Notifications appear in the Windows notification centre (bottom-right).

RESPONSE RULES:
- Keep responses SHORT. You are speaking aloud. 2-4 sentences max unless detail is essential.
- No markdown, no bullet points, no headers. Plain spoken English only.
- Never say "Certainly!", "Of course!", "Great!" — be natural, not a chatbot.
- Do not start every response with a greeting — just answer directly.

AVAILABLE ACTIONS - embed these tags in your reply when needed:
[ACTION:OPEN:<AppName>] - open any Windows app (Spotify, Chrome, Notepad, VS Code, Discord etc.)
[ACTION:WINDOW:<AppName>|<maximize|minimize|close|focus>] - control an open window
[ACTION:LIST_APPS:<filter>] - list installed apps
[ACTION:TERMINAL:<command>] - run a PowerShell command
[ACTION:SHORTCUT:<keys>] - send keyboard shortcut e.g. win+d, alt+tab
[ACTION:SEARCH:<query>] - web search, shows results panel
[ACTION:NEWS:<topic>] - latest news on a topic, shows panel
[ACTION:STOCK:<symbol>] - stock price e.g. AAPL, RELIANCE.NS, TCS.NS
[ACTION:FETCH:<url>] - fetch and read a webpage
[ACTION:SYSINFO] - CPU, RAM, disk, battery, uptime
[ACTION:PROCESS:list] - list running processes
[ACTION:PROCESS:kill:<n>] - kill a process
[ACTION:FILE:open:<path>] - open a file
[ACTION:FILE:list:<directory>] - list files in a directory
[ACTION:FILE:search:<n>] - search files by name
[ACTION:VOLUME:<0-100>] - set system volume
[ACTION:CLIPBOARD:get] - read clipboard
[ACTION:CLIPBOARD:set:<text>] - write to clipboard
[ACTION:NOTIFY:<title>|<message>] - Windows notification
[ACTION:CALENDAR] - Outlook calendar events
[ACTION:MAIL] - Outlook unread emails
[ACTION:NOTES:list] - list notes
[ACTION:NOTES:create:<title>|<body>] - create a note
[ACTION:REMEMBER:<key>|<value>] - store to memory
[ACTION:RECALL:<query>] - search memory
[ACTION:TASKS:add:<title>] - add task
[ACTION:TASKS:list] - list tasks
[ACTION:TASKS:done:<id>] - complete task

For Indian stocks add .NS suffix: RELIANCE.NS, TCS.NS, INFY.NS
Action tags are processed silently. Embed and continue naturally.
When you don't know something, say so briefly and honestly."""


# ── Action dispatcher ─────────────────────────────────────────────────────────
ACTION_RE = re.compile(r'\[ACTION:([^\]]+)\]')


async def dispatch_action(tag: str) -> str:
    """Process a single [ACTION:...] tag and return the result."""
    parts = tag.split(":", 1)
    action = parts[0].upper()
    arg = parts[1] if len(parts) > 1 else ""

    try:
        if action == "CALENDAR":
            from calendar_access import get_events, format_events_for_voice
            events = get_events()
            return format_events_for_voice(events)

        elif action == "MAIL":
            from mail_access import get_recent_messages, get_unread_count, format_messages_for_voice
            count = get_unread_count()
            msgs = get_recent_messages(5)
            prefix = f"You have {count} unread messages.\n"
            return prefix + format_messages_for_voice(msgs)

        elif action == "NOTES":
            sub = arg.split(":", 1)
            if sub[0] == "list":
                from notes_access import get_recent_notes
                notes = get_recent_notes(5)
                if not notes:
                    return "No notes found."
                return "\n".join(f"• {n['title'] or 'Untitled'}: {n['body'][:80]}" for n in notes)
            elif sub[0] == "create" and len(sub) > 1:
                title_body = sub[1].split("|", 1)
                title = title_body[0] if title_body else "SIA Note"
                body = title_body[1] if len(title_body) > 1 else ""
                from notes_access import create_note
                ok = create_note(title, body)
                return "Note created." if ok else "Could not create note."

        elif action == "OPEN":
            from actions import open_app
            return open_app(arg)

        elif action == "WINDOW":
            # WINDOW:<app>|<maximize|minimize|close|focus>
            from actions import control_window
            parts2 = arg.split("|", 1)
            app  = parts2[0].strip()
            verb = parts2[1].strip() if len(parts2) > 1 else "focus"
            return control_window(app, verb)

        elif action == "LIST_APPS":
            from actions import list_installed_apps
            apps = list_installed_apps(arg)
            if not apps:
                return "No matching apps found in Start Menu."
            return "Installed apps: " + ", ".join(apps[:20])

        elif action == "OPEN_URL":
            from actions import open_url_in_chrome
            return open_url_in_chrome(arg)

        elif action == "TERMINAL":
            from actions import run_terminal_command
            return run_terminal_command(arg)

        elif action == "SEARCH":
            from browser import search_web, format_search_results_for_voice
            results = await asyncio.get_event_loop().run_in_executor(
                None, search_web, arg
            )
            _pending_panel_data.append({
                "title": f"Search: {arg}",
                "type": "search",
                "items": results[:6],
            })
            return format_search_results_for_voice(results)

        elif action == "NEWS":
            from browser import get_news, format_news_for_voice
            topic = arg if arg else "top news today"
            items = await asyncio.get_event_loop().run_in_executor(
                None, get_news, topic
            )
            _pending_panel_data.append({
                "title": f"News: {topic}",
                "type": "search",
                "items": [{"title": n["title"], "url": n["url"],
                           "snippet": f"{n['source']}  {n['time']}"} for n in items],
            })
            return format_news_for_voice(items)

        elif action == "STOCK":
            from browser import get_stock, format_stock_for_voice
            symbol = arg.upper().strip()
            data = await asyncio.get_event_loop().run_in_executor(
                None, get_stock, symbol
            )
            if "error" not in data:
                _pending_panel_data.append({
                    "title": f"{data.get('name', symbol)} ({symbol})",
                    "type": "stock",
                    "fields": {
                        "Price":   data["price"],
                        "Change":  f"{data['change']} ({data['percent']})",
                        "High":    data["high"],
                        "Low":     data["low"],
                        "Volume":  data["volume"],
                        "Mkt Cap": data["mkt_cap"],
                    },
                })
            return format_stock_for_voice(data)

        elif action == "SYSINFO":
            from actions import get_system_info
            info = get_system_info()
            if "error" in info:
                return f"System info error: {info['error']}"
            _pending_panel_data.append({
                "title": "System Info",
                "type": "stock",
                "fields": {
                    "CPU":      info.get("cpu", "N/A"),
                    "RAM Used": info.get("ram_used", "N/A"),
                    "RAM Total":info.get("ram_total", "N/A"),
                    "Disk Free":info.get("disk_free", "N/A"),
                    "Battery":  f"{info.get('battery','N/A')} {info.get('batt_status','')}".strip(),
                    "Uptime":   info.get("uptime", "N/A"),
                },
            })
            return (f"CPU at {info.get('cpu','N/A')}, "
                    f"RAM {info.get('ram_used','N/A')} of {info.get('ram_total','N/A')} used, "
                    f"disk has {info.get('disk_free','N/A')} free, "
                    f"up for {info.get('uptime','N/A')}.")

        elif action == "PROCESS":
            sub = arg.split(":", 1)
            if sub[0] == "list":
                from actions import get_running_processes
                procs = get_running_processes(sub[1] if len(sub) > 1 else "", top=8)
                _pending_panel_data.append({
                    "title": "Running Processes",
                    "type": "search",
                    "items": [{"title": p["name"], "url": "",
                               "snippet": f"CPU: {p['cpu']}s  RAM: {p['mem_mb']} MB"} for p in procs],
                })
                return "Top processes: " + ", ".join(f"{p['name']} ({p['mem_mb']}MB)" for p in procs[:5])
            elif sub[0] == "kill" and len(sub) > 1:
                from actions import kill_process
                return kill_process(sub[1].strip())

        elif action == "FILE":
            sub = arg.split(":", 1)
            if sub[0] == "open" and len(sub) > 1:
                from actions import open_file
                return open_file(sub[1].strip())
            elif sub[0] == "list":
                from actions import list_files
                parts2 = sub[1].split("|") if len(sub) > 1 else []
                directory = parts2[0].strip() if parts2 else ""
                files = list_files(directory)
                _pending_panel_data.append({
                    "title": f"Files: {directory or 'Desktop'}",
                    "type": "search",
                    "items": [{"title": f, "url": "", "snippet": ""} for f in files],
                })
                return f"Found {len(files)} files: " + ", ".join(files[:5])
            elif sub[0] == "search" and len(sub) > 1:
                from actions import search_files
                results = search_files(sub[1].strip())
                _pending_panel_data.append({
                    "title": f"File search: {sub[1]}",
                    "type": "search",
                    "items": [{"title": r.split("\\")[-1], "url": "", "snippet": r} for r in results],
                })
                return f"Found {len(results)} files matching '{sub[1]}'." if results else f"No files found matching '{sub[1]}'."

        elif action == "SHORTCUT":
            from actions import send_keyboard_shortcut
            return send_keyboard_shortcut(arg)

        elif action == "VOLUME":
            from actions import set_volume
            try:
                level = int(arg)
                return set_volume(level)
            except ValueError:
                return "Invalid volume level."

        elif action == "CLIPBOARD":
            from actions import get_clipboard, set_clipboard
            if arg == "get":
                return get_clipboard() or "Clipboard is empty."
            elif arg.startswith("set:"):
                return set_clipboard(arg[4:])

        elif action == "NOTIFY":
            from actions import show_notification
            parts2 = arg.split("|", 1)
            title = parts2[0] if parts2 else "SIA"
            msg = parts2[1] if len(parts2) > 1 else ""
            return show_notification(title, msg)

        elif action == "REMEMBER":
            from memory import upsert_fact
            kv = arg.split("|", 1)
            if len(kv) == 2:
                upsert_fact(kv[0].strip(), kv[1].strip())
                return f"Remembered: {kv[0]}."
            return "Could not store fact."

        elif action == "RECALL":
            from memory import search_facts
            facts = search_facts(arg)
            if not facts:
                return "Nothing found in memory."
            return "\n".join(f"{f['key']}: {f['value']}" for f in facts)

        elif action == "TASKS":
            sub = arg.split(":", 1)
            if sub[0] == "list":
                from memory import get_tasks
                tasks = get_tasks(limit=10)
                if not tasks:
                    return "No tasks."
                return "\n".join(f"[{t['id']}] {t['title']} ({t['status']})" for t in tasks)
            elif sub[0] == "add" and len(sub) > 1:
                from memory import add_task
                tid = add_task(sub[1].strip())
                return f"Task added (id {tid})."
            elif sub[0] == "done" and len(sub) > 1:
                from memory import update_task_status
                update_task_status(int(sub[1]), "done")
                return "Task marked complete."

    except Exception as e:
        log.error(f"Action {action} error: {e}")
        return f"Action failed: {e}"

    return f"Unknown action: {action}"


async def process_actions(text: str) -> tuple[str, list[str]]:
    """Strip action tags from text, run them, return clean text + results."""
    tags = ACTION_RE.findall(text)
    clean = ACTION_RE.sub("", text).strip()
    clean = re.sub(r'\s+', ' ', clean)

    results = []
    for tag in tags:
        result = await dispatch_action(tag)
        if result:
            results.append(result)

    return clean, results


# ── TTS ───────────────────────────────────────────────────────────────────────
# Edge TTS voice — free, neural, no API key needed
# Options: en-IN-NeerjaNeural, en-US-AriaNeural, en-GB-SoniaNeural
EDGE_TTS_VOICE = os.getenv("EDGE_TTS_VOICE", "en-IN-NeerjaNeural")


async def tts_edge(text: str) -> bytes | None:
    """Microsoft Edge TTS — free neural voices, no API key, returns MP3 bytes."""
    try:
        import edge_tts
        communicate = edge_tts.Communicate(text, EDGE_TTS_VOICE)
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        if audio_chunks:
            return b"".join(audio_chunks)
    except ImportError:
        log.warning("edge-tts not installed — run: pip install edge-tts")
    except Exception as e:
        log.warning(f"Edge TTS failed: {e}")
    return None


async def tts_elevenlabs(text: str) -> bytes | None:
    """ElevenLabs TTS — premium option, requires valid API key."""
    if not ELEVENLABS_API_KEY:
        return None
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2_5",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    headers = {"xi-api-key": ELEVENLABS_API_KEY, "Content-Type": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(url, json=payload, headers=headers)
            if r.status_code == 200:
                return r.content
            else:
                log.warning(f"ElevenLabs TTS failed: {r.status_code} — {r.text[:200]}")
    except Exception as e:
        log.warning(f"ElevenLabs TTS exception: {e}")
    return None


async def synthesize_and_send(ws: WebSocket, text: str) -> None:
    """
    TTS priority:
    1. ElevenLabs (if key set and working)
    2. Edge TTS (free neural voice — primary fallback)
    3. Windows SAPI (last resort)
    """
    # Try ElevenLabs first if configured
    if ELEVENLABS_API_KEY:
        audio = await tts_elevenlabs(text)
        if audio:
            b64 = base64.b64encode(audio).decode()
            await ws.send_json({"type": "audio", "data": b64, "format": "mp3"})
            return

    # Try Edge TTS (free, neural, great quality)
    audio = await tts_edge(text)
    if audio:
        b64 = base64.b64encode(audio).decode()
        await ws.send_json({"type": "audio", "data": b64, "format": "mp3"})
        log.info("TTS: Edge neural voice used")
        return

    # Last resort: Windows SAPI (plays locally, no audio to browser)
    log.warning("TTS: Falling back to Windows SAPI")
    tts_windows_sapi(text)
    await ws.send_json({"type": "audio_local", "text": text})


# ── Echo filter ───────────────────────────────────────────────────────────────
_recent_sia_phrases: list[str] = []
MAX_ECHO_CACHE = 5


def is_echo(text: str) -> bool:
    """Detect if incoming speech is SIA's own TTS echo."""
    text_lower = text.lower().strip()
    for phrase in _recent_sia_phrases:
        if text_lower in phrase or phrase[:40] in text_lower:
            return True
    return False


def cache_sia_phrase(text: str):
    _recent_sia_phrases.append(text.lower().strip())
    if len(_recent_sia_phrases) > MAX_ECHO_CACHE:
        _recent_sia_phrases.pop(0)


# ── AI response ───────────────────────────────────────────────────────────────
async def get_ai_response(session_id: str, user_text: str) -> str:
    if not ai_async:
        return "I'm afraid my API key is not configured. Please add your Anthropic key to the .env file."

    session = get_session(session_id)
    history = session["history"]

    history.append({"role": "user", "content": user_text})
    trimmed = history[-20:]

    try:
        response = await ai_async.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=250,
            system=build_system_prompt(),
            messages=trimmed,
        )
        reply = response.content[0].text
        history.append({"role": "assistant", "content": reply})
        session["history"] = history
        return reply
    except anthropic.AuthenticationError:
        return "Authentication failed. Please check your Anthropic API key in the .env file."
    except anthropic.RateLimitError:
        return "I'm being rate-limited at the moment. Please try again shortly."
    except Exception as e:
        log.error(f"AI error: {type(e).__name__}: {e}")
        return f"I encountered a difficulty: {type(e).__name__}. Check the backend console."


# ── WebSocket voice handler ───────────────────────────────────────────────────
@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket):
    await ws.accept()
    session_id = str(uuid.uuid4())
    log.info(f"New voice session: {session_id}")

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "transcript":
                user_text = data.get("text", "").strip()
                if not user_text or len(user_text) < 2:
                    continue

                if is_echo(user_text):
                    log.debug(f"Echo filtered: {user_text[:50]}")
                    continue

                log.info(f"[{session_id[:8]}] User: {user_text}")

                # Notify frontend: thinking
                await ws.send_json({"type": "status", "state": "thinking"})

                try:
                    # Get AI response with timeout
                    raw_reply = await asyncio.wait_for(
                        get_ai_response(session_id, user_text),
                        timeout=30.0
                    )

                    # Process action tags
                    clean_reply, action_results = await process_actions(raw_reply)

                    # If actions returned data, do a second AI pass to voice it
                    if action_results:
                        context = "\n".join(action_results)
                        followup_prompt = (
                            f"Here is the data you retrieved:\n{context}\n\n"
                            f"Now respond to the user in 2-3 sentences, naturally, as SIA."
                        )
                        session = get_session(session_id)
                        session["history"].append({"role": "user", "content": followup_prompt})
                        final_reply = await asyncio.wait_for(
                            get_ai_response(session_id, followup_prompt),
                            timeout=20.0
                        )
                        final_clean, _ = await process_actions(final_reply)
                        clean_reply = final_clean or clean_reply

                    if not clean_reply:
                        clean_reply = "One moment."

                    log.info(f"[{session_id[:8]}] SIA: {clean_reply[:80]}")
                    cache_sia_phrase(clean_reply)

                    # Send text first (for display)
                    await ws.send_json({"type": "response", "text": clean_reply})

                    # Send any pending rich panel data
                    if _pending_panel_data:
                        for panel in _pending_panel_data:
                            await ws.send_json({"type": "panel_data", "panel": panel})
                        _pending_panel_data.clear()

                    # Send audio
                    await synthesize_and_send(ws, clean_reply)

                except asyncio.TimeoutError:
                    log.error(f"[{session_id[:8]}] AI response timed out")
                    await ws.send_json({"type": "response", "text": "I'm sorry, that took too long. Please try again."})
                    await ws.send_json({"type": "audio_local", "text": "I'm sorry, that took too long."})
                except Exception as e:
                    log.error(f"[{session_id[:8]}] Handler error: {type(e).__name__}: {e}")
                    await ws.send_json({"type": "response", "text": f"Error: {type(e).__name__} — check the backend console."})

                # Always return to listening
                await ws.send_json({"type": "status", "state": "listening"})

            elif msg_type == "ping":
                await ws.send_json({"type": "pong"})

            elif msg_type == "reset":
                if session_id in sessions:
                    sessions[session_id]["history"] = []
                await ws.send_json({"type": "status", "state": "reset"})

    except WebSocketDisconnect:
        log.info(f"Session disconnected: {session_id}")
    except Exception as e:
        log.error(f"WS fatal error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ── Live log streaming ────────────────────────────────────────────────────────
_log_buffer: list[dict] = []
_log_subscribers: list[WebSocket] = []
MAX_LOG_BUFFER = 200


_main_loop: asyncio.AbstractEventLoop | None = None


class WSLogHandler(logging.Handler):
    """Captures log records, buffers them, and pushes to WS subscribers."""

    def emit(self, record: logging.LogRecord):
        try:
            entry = {
                "ts": record.created,
                "level": record.levelname,
                "msg": self.format(record),
            }
            _log_buffer.append(entry)
            if len(_log_buffer) > MAX_LOG_BUFFER:
                _log_buffer.pop(0)
            # Push to connected subscribers thread-safely
            if _main_loop and _main_loop.is_running():
                for subscriber in list(_log_subscribers):
                    _main_loop.call_soon_threadsafe(
                        asyncio.ensure_future,
                        _safe_send(subscriber, entry)
                    )
        except Exception:
            pass


async def _safe_send(ws: WebSocket, entry: dict):
    try:
        await ws.send_json(entry)
    except Exception:
        if ws in _log_subscribers:
            _log_subscribers.remove(ws)


# Attach WS log handler to ROOT logger so uvicorn logs are captured too
_ws_log_handler = WSLogHandler()
_ws_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
logging.getLogger().addHandler(_ws_log_handler)
# Also capture uvicorn loggers
for _uvi_logger in ("uvicorn", "uvicorn.access", "uvicorn.error"):
    logging.getLogger(_uvi_logger).addHandler(_ws_log_handler)


@app.on_event("startup")
async def on_startup():
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    log.info("SIA backend ready ✓")


@app.websocket("/ws/logs")
async def logs_ws(ws: WebSocket):
    """Stream backend logs to the frontend terminal panel."""
    await ws.accept()
    _log_subscribers.append(ws)
    try:
        # Replay buffered logs first
        for entry in list(_log_buffer[-150:]):
            await ws.send_json(entry)
        # Keep connection alive waiting for client messages
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=25)
            except asyncio.TimeoutError:
                await ws.send_json({"level": "PING", "msg": "", "ts": time.time()})
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        if ws in _log_subscribers:
            _log_subscribers.remove(ws)


# ── REST API ──────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    return {
        "status": "ok",
        "version": "1.0.0",
        "ai_configured": bool(ANTHROPIC_API_KEY),
        "tts_configured": bool(ELEVENLABS_API_KEY),
        "active_sessions": len(sessions),
    }


@app.post("/api/wake")
async def wake():
    """Wake endpoint for double-clap or external triggers."""
    return JSONResponse({"status": "awake"})


@app.get("/api/memory/facts")
async def list_facts():
    from memory import all_facts
    return {"facts": all_facts()}


@app.post("/api/memory/facts")
async def add_fact(payload: dict):
    from memory import upsert_fact
    key = payload.get("key", "")
    value = payload.get("value", "")
    if not key or not value:
        raise HTTPException(400, "key and value required")
    upsert_fact(key, value)
    return {"status": "saved"}


@app.get("/api/tasks")
async def list_tasks():
    from memory import get_tasks
    return {"tasks": get_tasks()}


@app.post("/api/tasks")
async def create_task(payload: dict):
    from memory import add_task
    title = payload.get("title", "")
    if not title:
        raise HTTPException(400, "title required")
    tid = add_task(title, payload.get("description", ""))
    return {"id": tid}


@app.get("/api/calendar")
async def get_calendar():
    from calendar_access import get_events
    return {"events": get_events()}


@app.get("/api/mail")
async def get_mail():
    from mail_access import get_unread_count, get_recent_messages
    return {
        "unread": get_unread_count(),
        "messages": get_recent_messages(5),
    }


# ── Serve frontend dist (for production) ─────────────────────────────────────
dist_path = Path(__file__).parent / "frontend" / "dist"
if dist_path.exists():
    app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="static")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Start calendar background refresh
    try:
        from calendar_access import start_background_refresh
        start_background_refresh()
    except Exception:
        pass

    log.info("Starting SIA backend on http://localhost:8340")
    log.info("Frontend should be running on http://localhost:5173")

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8340,
        reload=False,
        log_level="info",
    )
