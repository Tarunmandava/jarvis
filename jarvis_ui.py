#!/usr/bin/env python3
"""
JARVIS — on-screen assistant using your own android photo.

Shows your image (jarvis.png) with a pulsing state-coloured aura and a soft
glow over her mouth while she speaks. The voice loop runs on a background
thread and drives the visuals.

Setup:
    pip install pillow
    Save your image as jarvis.png in this folder.
    python jarvis_ui.py

Reuses jarvis_voice.py (memory, reminders, tools). Say "hey jarvis".
"""

import os
import glob
import time
import queue
import threading
import tkinter as tk

import numpy as np
import sounddevice as sd
import openwakeword
from openwakeword.model import Model
import anthropic
from PIL import Image, ImageTk

from jarvis_voice import (listen, speak, ask, answer_text,
                          load_memory, save_memory, due_reminders)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
IMAGE_FILE = "jarvis.png"        # your android picture, in this folder
IMG_WIDTH = 160                  # toy size; raise to 380 for a big version
SHOW_CAPTIONS = False            # True to show the you/jarvis text under her
MOUTH_REL = (0.50, 0.40)         # where her mouth is, as (x, y) fractions of
                                 # the image — nudge until the speaking glow
                                 # sits on her lips

SR = 16000
CHUNK = 1280
THRESHOLD = 0.4
WAKE = "hey_jarvis"

ui_queue = queue.Queue()


def ui(kind, value=""):
    ui_queue.put((kind, value))


# --------------------------------------------------------------------------- #
# Background voice loop
# --------------------------------------------------------------------------- #
def wait_for_wake(oww):
    oww.reset()
    last = time.time()
    with sd.InputStream(samplerate=SR, channels=1, dtype="int16",
                        blocksize=CHUNK) as stream:
        while True:
            frame, _ = stream.read(CHUNK)
            if max(oww.predict(frame[:, 0]).values()) >= THRESHOLD:
                return None
            if time.time() - last > 1.0:
                last = time.time()
                fired = due_reminders()
                if fired:
                    return fired


def worker():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        ui("state", "error")
        ui("jarvis", "Set ANTHROPIC_API_KEY first.")
        return

    ui("jarvis", "Booting\u2026")
    openwakeword.utils.download_models()
    oww = Model(wakeword_models=[WAKE], inference_framework="onnx")
    client = anthropic.Anthropic()
    memory = load_memory()
    messages = [dict(t) for t in memory]

    ui("state", "idle")
    ui("jarvis", "")
    while True:
        try:
            fired = wait_for_wake(oww)
            if fired:
                for text in fired:
                    ui("state", "speaking")
                    ui("jarvis", f"Reminder: {text}")
                    speak(f"Reminder: {text}")
                ui("state", "idle")
                continue

            ui("state", "listening")
            ui("you", "\u2026")
            user = listen()
            if not user:
                ui("state", "idle")
                ui("jarvis", "(didn't catch that)")
                speak("I didn't catch that, sir.")
                continue

            ui("you", user)
            ui("state", "thinking")
            messages.append({"role": "user", "content": user})
            reply = answer_text(ask(client, messages).content)

            ui("jarvis", reply)
            ui("state", "speaking")
            speak(reply)

            memory.append({"role": "user", "content": user})
            memory.append({"role": "assistant", "content": reply})
            save_memory(memory)
            ui("state", "idle")

        except Exception as e:
            ui("state", "idle")
            ui("jarvis", f"error: {e}")


# --------------------------------------------------------------------------- #
# The window
# --------------------------------------------------------------------------- #
BG = "#ff00ff"          # magic transparent colour (you'll never see it)
STATE_COLOR = {"idle": "#5b6b93", "listening": "#22d3ee",
               "thinking": "#f59e0b", "speaking": "#34d399", "error": "#ef4444"}
STATE_TEXT = {"idle": "asleep \u2014 say \u201chey jarvis\u201d", "listening": "listening\u2026",
              "thinking": "thinking\u2026", "speaking": "speaking\u2026", "error": "error"}


class JarvisUI:
    def __init__(self, root):
        self.root = root
        self.state = "idle"
        self.phase = 0.0
        self.bg_img = None
        self.W = IMG_WIDTH
        self.H = 300

        root.title("Miki")
        root.configure(bg=BG)
        root.attributes("-topmost", True)
        root.overrideredirect(True)              # no title bar / border
        root.attributes("-transparentcolor", BG)  # BG colour becomes see-through
        # drag to move (no title bar to grab), Esc to close
        root.bind("<Button-1>", self._start_drag)
        root.bind("<B1-Motion>", self._on_drag)
        root.bind("<Escape>", lambda e: root.destroy())
        self._dx = self._dy = 0

        # Find an image regardless of exact name/extension (Windows often
        # hides extensions, so "jarvis.png" can really be "jarvis.png.jpg").
        folder = os.path.dirname(os.path.abspath(__file__))
        candidates = []
        exact = os.path.join(folder, IMAGE_FILE)
        if os.path.exists(exact):
            candidates.append(exact)
        for pat in ("jarvis.*", "*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
            for h in glob.glob(os.path.join(folder, pat)):
                if h.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp")):
                    candidates.append(h)

        for path in candidates:
            try:
                img = Image.open(path).convert("RGBA")
                # transparent areas -> magic colour -> become see-through
                canvas_bg = Image.new("RGBA", img.size, (255, 0, 255, 255))
                canvas_bg.paste(img, (0, 0), img)
                img = canvas_bg.convert("RGB")
                self.H = int(img.height * self.W / img.width)
                self.bg_img = ImageTk.PhotoImage(img.resize((self.W, self.H)))
                break
            except Exception:
                continue

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        wh = self.H + (120 if SHOW_CAPTIONS else 0)
        root.geometry(f"{self.W}x{wh}+{sw - self.W - 30}+{sh - wh - 70}")

        self.canvas = tk.Canvas(root, width=self.W, height=self.H, bg=BG,
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.tag_bind("close", "<Button-1>", lambda e: self.root.destroy())
        self.status = self.you = self.jarvis = None
        if SHOW_CAPTIONS:
            self.status = tk.Label(root, text=STATE_TEXT["idle"], fg="#8b97b5",
                                   bg=BG, font=("Segoe UI", 10))
            self.status.pack()
            self.you = tk.Label(root, text="", fg="#cbd5e1", bg=BG,
                                font=("Segoe UI", 10), wraplength=self.W - 30,
                                justify="center")
            self.you.pack(pady=(8, 0))
            self.jarvis = tk.Label(root, text="", fg="#e2e8f0", bg=BG,
                                   font=("Segoe UI", 11, "bold"),
                                   wraplength=self.W - 30, justify="center")
            self.jarvis.pack(pady=(2, 0))

        self.animate()
        self.poll()

    def _start_drag(self, e):
        self._dx, self._dy = e.x, e.y

    def _on_drag(self, e):
        x = self.root.winfo_x() + e.x - self._dx
        y = self.root.winfo_y() + e.y - self._dy
        self.root.geometry(f"+{x}+{y}")

    def animate(self):
        self.phase += 0.15
        c = self.canvas
        c.delete("all")
        color = STATE_COLOR.get(self.state, "#5b6b93")
        s = abs(np.sin(self.phase))

        if self.bg_img:
            c.create_image(0, 0, anchor="nw", image=self.bg_img)
        else:
            c.create_text(self.W / 2, self.H / 2, fill="#e2e8f0",
                          font=("Segoe UI", 11),
                          text="Put your image here as\njarvis.png, then restart.")

        # pulsing aura frame
        gw = 5 + 4 * s
        c.create_rectangle(gw / 2, gw / 2, self.W - gw / 2, self.H - gw / 2,
                           outline=color, width=gw)

        # mouth glow while speaking
        if self.state == "speaking":
            mx, my = self.W * MOUTH_REL[0], self.H * MOUTH_REL[1]
            rr = 10 + 12 * abs(np.sin(self.phase * 2.5))
            c.create_oval(mx - rr, my - rr * 0.6, mx + rr, my + rr * 0.6,
                          outline="", fill=color, stipple="gray25")

        # small close button (top-right) — click to shut Miki down
        bx, by = self.W - 14, 14
        c.create_oval(bx - 9, by - 9, bx + 9, by + 9, fill="#c0392b",
                      outline="#ffffff", width=1, tags="close")
        c.create_line(bx - 4, by - 4, bx + 4, by + 4, fill="#ffffff",
                      width=2, tags="close")
        c.create_line(bx - 4, by + 4, bx + 4, by - 4, fill="#ffffff",
                      width=2, tags="close")

        self.root.after(40, self.animate)

    def poll(self):
        try:
            while True:
                kind, value = ui_queue.get_nowait()
                if kind == "state":
                    self.state = value
                    if self.status:
                        self.status.config(text=STATE_TEXT.get(value, value))
                elif kind == "you":
                    if self.you:
                        self.you.config(text="" if value == "\u2026" else f"you: {value}")
                elif kind == "jarvis":
                    if self.jarvis:
                        self.jarvis.config(text=f"jarvis: {value}" if value else "")
        except queue.Empty:
            pass
        self.root.after(50, self.poll)


def main():
    root = tk.Tk()
    JarvisUI(root)
    threading.Thread(target=worker, daemon=True).start()
    root.mainloop()


if __name__ == "__main__":
    main()