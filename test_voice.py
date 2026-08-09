#!/usr/bin/env python3
"""
Isolated voice check — run BEFORE wiring voice into the full assistant.
Confirms the mouth (TTS) and ears (mic + STT) work on their own, so any
audio problem is caught here instead of tangled up with the brain/tools.

    python test_voice.py
"""

import numpy as np
import sounddevice as sd
import pyttsx3
from faster_whisper import WhisperModel

SR = 16000

# ---- 1) Mouth: text-to-speech --------------------------------------------
print("1/2  Testing text-to-speech — you should hear a voice now...")
e = pyttsx3.init()
e.say("Voice check. If you can hear this, the mouth works.")
e.runAndWait()
e.stop()
print("     ...spoken. Did you hear it?\n")

# ---- 2) Ears: microphone + speech-to-text --------------------------------
SECONDS = 4
print(f"2/2  Testing the microphone — speak for ~{SECONDS} seconds after the prompt.")
e = pyttsx3.init()
e.say("Please speak now.")
e.runAndWait()
e.stop()

audio = sd.rec(int(SECONDS * SR), samplerate=SR, channels=1, dtype="float32")
sd.wait()
audio = audio[:, 0]

print("     transcribing (first run downloads the model, be patient)...")
model = WhisperModel("base.en", device="cpu", compute_type="int8")
segments, _ = model.transcribe(audio, language="en")
heard = " ".join(s.text.strip() for s in segments).strip()

print(f"\n     I heard: {heard!r}")
print("\nIf the voice spoke AND the transcript is close to what you said,")
print("both subsystems work — you're clear to wire voice into the loop.")