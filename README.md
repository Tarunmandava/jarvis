# Miki — a voice assistant built from scratch

A wake-word voice assistant for Windows, written in Python. Say the wake word
and she listens, thinks, and talks back — with an animated on-screen face.

## What she does
- **Wake word** (openWakeWord) — runs locally, no account
- **Speech-to-text** on-device with faster-whisper
- **Reasoning** via the Claude API, with live **web search**
- **Actions**: opens files, apps, and websites by voice
- **Productivity**: drafts emails, tracks job applications, sets reminders
- **Memory**: remembers the conversation across restarts
- **Voice**: natural neural TTS (edge-tts)
- **Face**: a small always-on-top animated avatar that reacts as she listens,
  thinks, and speaks

## Stack
Python · Claude API · faster-whisper · openWakeWord · edge-tts · Tkinter · Pillow

## Setup
1. `pip install -r requirements.txt`
2. Set your Anthropic key: `setx ANTHROPIC_API_KEY "your-key"` (reopen terminal)
3. Add an avatar image named `jarvis.png` in the folder
4. Run: `python jarvis_ui.py`  (or `python jarvis_wake_oww.py` for no window)

Say **"hey jarvis"** to wake her.

## Files
- `jarvis_voice.py` — the brain, tools, memory, voice
- `jarvis_wake_oww.py` — wake-word loop (headless)
- `jarvis_ui.py` — the animated on-screen face
- `test_voice.py` — quick mic + speaker check

Built as a personal learning project.
