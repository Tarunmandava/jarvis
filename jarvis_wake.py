#!/usr/bin/env python3
"""
JARVIS — always-listening wake-word layer.

Sits quietly until it hears "Jarvis", then runs one voice turn using the
functions from jarvis_voice.py, then goes back to sleep. Your working
jarvis_voice.py is imported unchanged.

Setup:
    pip install pvporcupine pvrecorder
    setx PICOVOICE_ACCESS_KEY "your-free-key"   # from https://console.picovoice.ai
    (ANTHROPIC_API_KEY must already be set from before)
    python jarvis_wake.py

Then just say "Jarvis" out loud. Ctrl+C to quit.
"""

import os
import sys

import pvporcupine
from pvrecorder import PvRecorder
import anthropic

# Reuse everything you already built and tested.
from jarvis_voice import listen, speak, ask, answer_text

PICOVOICE_KEY = os.environ.get("PICOVOICE_ACCESS_KEY")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")
    if not PICOVOICE_KEY:
        sys.exit("Set PICOVOICE_ACCESS_KEY first (free key at https://console.picovoice.ai).")

    client = anthropic.Anthropic()
    messages = []

    # "jarvis" is a built-in Porcupine keyword — no training needed.
    porcupine = pvporcupine.create(access_key=PICOVOICE_KEY, keywords=["jarvis"])
    recorder = PvRecorder(frame_length=porcupine.frame_length)
    recorder.start()

    print('Jarvis is asleep. Say "Jarvis" to wake it.  (Ctrl+C to quit)')

    try:
        while True:
            pcm = recorder.read()
            if porcupine.process(pcm) >= 0:           # heard the wake word
                print("\n(woken)")
                recorder.stop()                       # release mic for recording
                speak("Yes?")
                user = listen()                       # capture the actual command

                if user:
                    print(f"you  \u203a {user}")
                    messages.append({"role": "user", "content": user})
                    reply = answer_text(ask(client, messages).content)
                    print(f"jarvis \u203a {reply}\n")
                    speak(reply)

                recorder.start()                      # back to sleep, listening again
                print('...asleep. Say "Jarvis" again.')
    except KeyboardInterrupt:
        speak("Powering down.")
    finally:
        recorder.delete()
        porcupine.delete()


if __name__ == "__main__":
    main()