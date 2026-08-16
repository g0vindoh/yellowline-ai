"""
YellowLine AI — Audio Backend v8.0

Cross-platform audio: WAV prompts → pyttsx3 TTS → espeak → silent.
Buzzer: sounddevice → simpleaudio → aplay → winsound → silent.
"""

import os
import threading
import subprocess
import wave

_tts_lock   = threading.Lock()
_tts_engine = None

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import pyttsx3
    _tts_engine = pyttsx3.init()
    _tts_engine.setProperty("rate", 160)
    _tts_engine.setProperty("volume", 1.0)
    PYTTSX3_OK = True
except Exception:
    PYTTSX3_OK = False

try:
    import sounddevice as sd
    import scipy.io.wavfile as _wav
    SOUNDDEVICE_OK = True
except Exception:
    SOUNDDEVICE_OK = False

try:
    import simpleaudio as sa
    SIMPLEAUDIO_OK = True
except Exception:
    SIMPLEAUDIO_OK = False

AUDIO_PROMPTS_DIR = os.environ.get("YL_AUDIO_PROMPTS", "audio_prompts")
LANG              = os.environ.get("YL_LANG", "en")

# ── WAV prompt lookup ─────────────────────────────────────────────────────────
_PROMPT_FILES = {
    "system_online":   "system_online",
    "approach_edge":   "approach_edge",
    "critical_edge":   "critical_edge",
    "fall":            "fall",
    "surge":           "surge",
    "edge_loss":       "edge_loss",
}

def _find_wav(phrase_key):
    if not phrase_key:
        return None
    for lang in (LANG, "en"):
        path = os.path.join(AUDIO_PROMPTS_DIR, f"{_PROMPT_FILES.get(phrase_key, phrase_key)}_{lang}.wav")
        if os.path.exists(path):
            return path
    return None


def _play_wav(path):
    if SOUNDDEVICE_OK:
        try:
            sr, data = _wav.read(path)
            sd.play(data, sr)
            sd.wait()
            return True
        except Exception:
            pass
    if SIMPLEAUDIO_OK:
        try:
            sa.WaveObject.from_wave_file(path).play().wait_done()
            return True
        except Exception:
            pass
    try:
        subprocess.run(["aplay", "-q", path],
                       timeout=10, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except Exception:
        pass
    try:
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
        return True
    except Exception:
        pass
    return False


def speak_text(text, phrase_key=None):
    """Play pre-rendered WAV if available, otherwise TTS, otherwise espeak."""
    def _speak():
        with _tts_lock:
            wav = _find_wav(phrase_key)
            if wav and _play_wav(wav):
                return
            if PYTTSX3_OK and _tts_engine:
                try:
                    _tts_engine.say(text)
                    _tts_engine.runAndWait()
                    return
                except Exception:
                    pass
            try:
                lang_flag = ["-v", "kn"] if LANG == "kn" else ["-v", "en"]
                subprocess.run(["espeak"] + lang_flag + [text],
                               timeout=10,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            except Exception:
                pass

    threading.Thread(target=_speak, daemon=True).start()


def play_buzzer(alarm_file):
    """Play the alarm WAV via best available backend."""
    def _play():
        if _play_wav(alarm_file):
            return
    threading.Thread(target=_play, daemon=True).start()


def make_alarm(path):
    """Generate a two-tone alarm WAV if it doesn't already exist."""
    if os.path.exists(path):
        return
    try:
        import numpy as np
        sr, dur, vol = 44100, 0.6, 32000
        n  = int(sr * dur)
        t  = __import__("numpy").linspace(0, dur, n, False)
        h  = n // 2
        wd = __import__("numpy").concatenate([
            __import__("numpy").sin(2 * 3.14159 * 880  * t[:h]),
            __import__("numpy").sin(2 * 3.14159 * 1200 * t[h:])
        ])
        fd = int(sr * 0.01)
        wd[:fd]  *= __import__("numpy").linspace(0, 1, fd)
        wd[-fd:] *= __import__("numpy").linspace(1, 0, fd)
        wd = (wd * vol).astype(__import__("numpy").int16)
        with wave.open(path, "w") as wf:
            wf.setnchannels(1); wf.setsampwidth(2)
            wf.setframerate(sr); wf.writeframes(wd.tobytes())
    except Exception as e:
        print(f"[Audio] Could not generate alarm.wav: {e}")
