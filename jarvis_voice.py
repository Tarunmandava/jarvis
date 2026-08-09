#!/usr/bin/env python3
"""
JARVIS — voice assistant with web search, file opening, browser, PC control,
and a job-application tracker. Brain is cloud (Claude); ears/mouth are local.

Setup:
    pip install anthropic faster-whisper sounddevice numpy pyttsx3
    setx ANTHROPIC_API_KEY "sk-ant-..."   # reopen the terminal after this
    python jarvis_wake_oww.py

SAFETY: run_command prints the exact command and waits for you to type 'y'.
"""

import os
import sys
import json
import time
import datetime
import subprocess
import webbrowser
import urllib.parse

import numpy as np
import sounddevice as sd
import pyttsx3
import anthropic
from faster_whisper import WhisperModel

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
MODEL = "claude-haiku-4-5-20251001"
USER_TITLE = "sir"
SR = 16000
MAX_TOKENS = 350

HOME = os.path.expanduser("~")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
JOBS_FILE = os.path.join(BASE_DIR, "jobs.json")
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")
REMINDERS_FILE = os.path.join(BASE_DIR, "reminders.json")
MAX_TURNS = 20                   # past turns remembered across restarts
VOICE_HINT = "zira"              # fallback Windows voice (substring); "" = default
SPEAK_RATE = 180                 # fallback voice speed
VOICE = "en-IN-NeerjaNeural"     # edge-tts neural voice (Indian English, female)
SEARCH_DIRS = [os.path.join(HOME, d) for d in ("Desktop", "Documents", "Downloads")]

PERSONA = f"""You are Miki, a friendly voice assistant on the user's Windows PC.
Dry, concise, competent, never sycophantic. Address the user as "{USER_TITLE}"
occasionally. Replies are spoken aloud: answer in ONE or TWO short sentences,
no markdown or lists unless asked.

Tools you have:
- open_file: when the user says "open <something>" and means a file or document.
- open_url: open a website or link. Build the correct URL yourself — e.g. for
  "play lofi on youtube" open https://www.youtube.com/results?search_query=lofi,
  for "open my linkedin" open https://www.linkedin.com/feed/.
- open_app: launch a Windows app like notepad or calc.
- draft_email: when the user wants to email or message someone, write the FULL,
  well-phrased email yourself and put it in the body; open a Gmail draft for
  review. If you don't know the recipient's address, leave 'to' empty and say so.
  Keep your SPOKEN reply short — the email text goes in the tool, not aloud.
- set_reminder: when the user asks to be reminded in N minutes/hours, convert to minutes.
- add_job / list_jobs / update_job: track the user's job applications.
- web_search: only for genuinely current info; otherwise answer directly.
- run_command: last resort; the user must confirm it."""

TOOLS = [
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 2},
    {"name": "open_file",
     "description": "Find and open a file on the PC by name. Searches Desktop, Documents, Downloads.",
     "input_schema": {"type": "object",
                      "properties": {"query": {"type": "string", "description": "words from the file name, e.g. 'resume' or 'fish dataset'"}},
                      "required": ["query"]}},
    {"name": "open_url",
     "description": "Open a URL in the browser. Construct search or site URLs as needed (YouTube, LinkedIn, etc.).",
     "input_schema": {"type": "object",
                      "properties": {"url": {"type": "string"}}, "required": ["url"]}},
    {"name": "open_app",
     "description": "Open a Windows app (e.g. 'notepad', 'calc').",
     "input_schema": {"type": "object",
                      "properties": {"name": {"type": "string"}}, "required": ["name"]}},
    {"name": "draft_email",
     "description": "Open a pre-filled Gmail draft for the user to review and send.",
     "input_schema": {"type": "object",
                      "properties": {"to": {"type": "string", "description": "recipient email if known, else empty"},
                                     "subject": {"type": "string"},
                                     "body": {"type": "string", "description": "the full email text"}},
                      "required": ["subject", "body"]}},
    {"name": "set_reminder",
     "description": "Remind the user after a number of minutes. Convert 'in 2 hours' to minutes=120.",
     "input_schema": {"type": "object",
                      "properties": {"minutes": {"type": "number"}, "text": {"type": "string"}},
                      "required": ["minutes", "text"]}},
    {"name": "add_job",
     "description": "Record a new job application.",
     "input_schema": {"type": "object",
                      "properties": {"company": {"type": "string"},
                                     "role": {"type": "string"},
                                     "status": {"type": "string", "description": "e.g. applied, interview, offer, rejected"}},
                      "required": ["company"]}},
    {"name": "list_jobs",
     "description": "List tracked job applications, optionally filtered by status.",
     "input_schema": {"type": "object",
                      "properties": {"status": {"type": "string"}}, "required": []}},
    {"name": "update_job",
     "description": "Update the status of a tracked application by company name.",
     "input_schema": {"type": "object",
                      "properties": {"company": {"type": "string"}, "status": {"type": "string"}},
                      "required": ["company", "status"]}},
    {"name": "run_command",
     "description": "Run a shell command. The user must confirm before it executes.",
     "input_schema": {"type": "object",
                      "properties": {"command": {"type": "string"}}, "required": ["command"]}},
]

# --------------------------------------------------------------------------- #
# Ears / mouth
# --------------------------------------------------------------------------- #
_whisper = WhisperModel("base.en", device="cpu", compute_type="int8")


def listen(max_s=15, thresh=0.015, silence_s=1.2, min_speech_s=0.4):
    """Record until you stop talking; won't stop until real speech is heard."""
    frames, silent, speech = [], 0, 0
    need_silence = int(silence_s / 0.1)
    need_speech = int(min_speech_s / 0.1)
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32") as stream:
        for _ in range(int(max_s / 0.1)):
            chunk = stream.read(int(SR * 0.1))[0][:, 0]
            frames.append(chunk)
            if np.sqrt(np.mean(chunk ** 2)) > thresh:
                speech += 1
                silent = 0
            elif speech >= need_speech:
                silent += 1
                if silent >= need_silence:
                    break
    audio = np.concatenate(frames) if frames else np.zeros(0, "float32")
    if audio.size == 0 or speech < need_speech:
        return ""
    segments, _ = _whisper.transcribe(audio, language="en")
    return " ".join(s.text.strip() for s in segments).strip()


# Natural neural voice (edge-tts). Falls back to the Windows voice if the
# libraries aren't installed or there's no internet.
try:
    import asyncio
    import tempfile
    import edge_tts
    import soundfile as sf
    _EDGE_OK = True
except Exception:
    _EDGE_OK = False


def _pyttsx_speak(text):
    engine = pyttsx3.init()
    if VOICE_HINT:
        for v in engine.getProperty("voices"):
            if VOICE_HINT.lower() in v.name.lower():
                engine.setProperty("voice", v.id)
                break
    engine.setProperty("rate", SPEAK_RATE)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def speak(text):
    if not text:
        return
    if _EDGE_OK:
        try:
            path = os.path.join(tempfile.gettempdir(), "jarvis_tts.mp3")
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(edge_tts.Communicate(text, VOICE).save(path))
            finally:
                loop.close()
            data, sr = sf.read(path, dtype="float32")
            sd.play(data, sr)
            sd.wait()
            return
        except Exception:
            pass                      # fall back to the offline Windows voice
    _pyttsx_speak(text)


# --------------------------------------------------------------------------- #
# Hands
# --------------------------------------------------------------------------- #
def _open_in_browser(url):
    for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
        if os.path.exists(p):
            try:
                webbrowser.get(f'"{p}" %s').open(url)
                return
            except Exception:
                break
    webbrowser.open(url)


def _find_files(query, limit=8):
    words = query.lower().split()
    hits = []
    for base in SEARCH_DIRS:
        if not os.path.isdir(base):
            continue
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs
                       if not d.startswith(".") and d.lower() not in
                       ("node_modules", "__pycache__", ".git", "venv", ".venv")]
            for f in files:
                if all(w in f.lower() for w in words):
                    hits.append(os.path.join(root, f))
            if len(hits) >= 40:
                break
    hits.sort(key=lambda p: len(os.path.basename(p)))
    return hits[:limit]


def _load_jobs():
    try:
        with open(JOBS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_jobs(jobs):
    with open(JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)


def load_memory():
    """Past conversation turns, so Jarvis remembers across restarts."""
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_memory(turns):
    with open(MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(turns[-MAX_TURNS * 2:], f, indent=2)


def _load_reminders():
    try:
        with open(REMINDERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_reminders(rem):
    with open(REMINDERS_FILE, "w", encoding="utf-8") as f:
        json.dump(rem, f, indent=2)


def due_reminders():
    """Texts of any reminders now due; also drops them from the store."""
    rem = _load_reminders()
    now = time.time()
    due = [r["text"] for r in rem if r["due"] <= now]
    if due:
        _save_reminders([r for r in rem if r["due"] > now])
    return due


def execute_tool(name, args):
    if name == "open_file":
        hits = _find_files(args["query"])
        if not hits:
            return f"I couldn't find a file matching '{args['query']}', sir."
        os.startfile(hits[0])
        extra = f" I found {len(hits)}; opened the closest." if len(hits) > 1 else ""
        return f"Opening {os.path.basename(hits[0])}.{extra}"

    if name == "open_url":
        _open_in_browser(args["url"])
        return "Opening that in the browser."

    if name == "open_app":
        try:
            os.startfile(args["name"])
            return f"Opened {args['name']}."
        except Exception:
            subprocess.Popen(["cmd", "/c", "start", "", args["name"]])
            return f"Attempted to open {args['name']}."

    if name == "draft_email":
        params = urllib.parse.urlencode({
            "view": "cm", "fs": "1",
            "to": args.get("to", ""),
            "su": args.get("subject", ""),
            "body": args.get("body", ""),
        })
        _open_in_browser("https://mail.google.com/mail/?" + params)
        print("\n--- draft ---")
        print("To:", args.get("to", "(you fill in)"))
        print("Subject:", args.get("subject", ""))
        print(args.get("body", ""))
        print("-------------\n")
        return "I've opened a draft in Gmail for you to review, sir."

    if name == "set_reminder":
        rem = _load_reminders()
        rem.append({"text": args["text"],
                    "due": time.time() + float(args["minutes"]) * 60})
        _save_reminders(rem)
        mins = int(float(args["minutes"]))
        return f"Reminder set for {mins} minute{'s' if mins != 1 else ''} from now."

    if name == "add_job":
        jobs = _load_jobs()
        jobs.append({"company": args["company"], "role": args.get("role", ""),
                     "status": args.get("status", "applied"),
                     "date": datetime.date.today().isoformat()})
        _save_jobs(jobs)
        return f"Logged {args.get('role', 'a role')} at {args['company']}."

    if name == "list_jobs":
        jobs = _load_jobs()
        status = args.get("status")
        if status:
            jobs = [j for j in jobs if j["status"].lower() == status.lower()]
        if not jobs:
            return "No applications tracked yet, sir."
        lines = [f"{j['role'] or 'a role'} at {j['company']}, {j['status']}" for j in jobs]
        return f"{len(jobs)} applications. " + "; ".join(lines) + "."

    if name == "update_job":
        jobs = _load_jobs()
        for j in jobs:
            if args["company"].lower() in j["company"].lower():
                j["status"] = args["status"]
                _save_jobs(jobs)
                return f"Updated {j['company']} to {args['status']}."
        return f"I don't have an application at {args['company']}, sir."

    if name == "run_command":
        cmd = args["command"]
        print(f"\n\u26a0  Jarvis wants to run:\n    {cmd}")
        if input("   allow? [y/N] ").strip().lower() != "y":
            return "User declined to run the command."
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)
        return (r.stdout + r.stderr)[:2000] or "Done."

    return f"Unknown tool: {name}"


# --------------------------------------------------------------------------- #
# Brain
# --------------------------------------------------------------------------- #
def ask(client, messages):
    while True:
        resp = client.messages.create(
            model=MODEL, max_tokens=MAX_TOKENS, system=PERSONA,
            messages=messages, tools=TOOLS,
        )
        messages.append({"role": "assistant", "content": resp.content})
        if resp.stop_reason == "pause_turn":
            continue
        if resp.stop_reason == "tool_use":
            results = [{"type": "tool_result", "tool_use_id": b.id,
                        "content": execute_tool(b.name, b.input)}
                       for b in resp.content if b.type == "tool_use"]
            messages.append({"role": "user", "content": results})
            continue
        return resp


def answer_text(content_blocks):
    return "".join(b.text for b in content_blocks if b.type == "text").strip()


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")
    client = anthropic.Anthropic()
    messages = []
    print("JARVIS online. Speak when ready (Ctrl+C to quit).\n")
    speak(f"Online and ready, {USER_TITLE}.")
    while True:
        try:
            print("listening...")
            user = listen()
            if not user:
                continue
            print(f"you  \u203a {user}")
            if user.lower().strip(" .") in {"exit", "quit", "goodbye", "shut down"}:
                speak(f"Powering down. Goodbye, {USER_TITLE}.")
                break
            messages.append({"role": "user", "content": user})
            reply = answer_text(ask(client, messages).content)
            print(f"jarvis \u203a {reply}\n")
            speak(reply)
        except KeyboardInterrupt:
            speak("Powering down.")
            break
        except anthropic.APIError as e:
            print(f"API error: {e}")


if __name__ == "__main__":
    main()