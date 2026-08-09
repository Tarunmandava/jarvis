#!/usr/bin/env python3
"""
JARVIS — always-listening wake word (openWakeWord), now with memory + reminders.

Sits quietly until it hears "hey jarvis". Remembers past conversation across
restarts, and speaks reminders when they come due. Fully local wake word.

Setup:
    pip install openwakeword
    (ANTHROPIC_API_KEY must already be set)
    python jarvis_wake_oww.py
"""

import os
import sys
import time

import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model
import anthropic

# Reuse everything built in jarvis_voice.py
from jarvis_voice import (listen, speak, ask, answer_text,
                          load_memory, save_memory, due_reminders)

SR = 16000
CHUNK = 1280
THRESHOLD = 0.4       # raise if it wakes on noise; lower if it misses you
WAKE = "hey_jarvis"
DEBUG = True          # set False for the invisible startup version


def wait_for_wake(oww):
    """
    Block until the wake word is heard OR a reminder comes due.
    Returns a list of reminder texts if any fired, else None (woke normally).
    """
    oww.reset()
    last_check = time.time()
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                        blocksize=CHUNK) as stream:
        while True:
            frame, _ = stream.read(CHUNK)
            score = max(oww.predict(frame[:, 0]).values())
            if DEBUG and score > 0.1:
                print(f"  heard something... score={score:.2f}", end="\r")
            if score >= THRESHOLD:
                return None
            # check reminders about once a second
            if time.time() - last_check > 1.0:
                last_check = time.time()
                fired = due_reminders()
                if fired:
                    return fired
    # stream closes here, freeing the mic


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Set ANTHROPIC_API_KEY first.")

    print("Loading wake-word model (first run downloads it)...")
    openwakeword.utils.download_models()
    oww = Model(wakeword_models=[WAKE], inference_framework="onnx")

    client = anthropic.Anthropic()
    memory = load_memory()                      # remember across restarts
    messages = [dict(t) for t in memory]        # seed the conversation

    print('Jarvis is asleep. Say "hey jarvis" to wake it.  (Ctrl+C to quit)')

    while True:
        try:
            fired = wait_for_wake(oww)

            if fired:                            # a reminder came due
                for text in fired:
                    print(f"\n(reminder) {text}")
                    speak(f"Reminder: {text}")
                continue

            print("\n(woken)")
            speak("Yes?")
            user = listen()

            if not user:
                print("  (didn't catch a command)")
                speak("I didn't catch that, sir.")
                continue

            print(f"you  \u203a {user}")
            messages.append({"role": "user", "content": user})
            reply = answer_text(ask(client, messages).content)
            print(f"jarvis \u203a {reply}\n")
            speak(reply)

            # persist this turn so it survives a restart
            memory.append({"role": "user", "content": user})
            memory.append({"role": "assistant", "content": reply})
            save_memory(memory)

        except KeyboardInterrupt:
            speak("Powering down.")
            break
        except Exception as e:
            print(f"  error this turn: {e}")
            try:
                speak("Something went wrong, sir.")
            except Exception:
                pass


if __name__ == "__main__":
    main()