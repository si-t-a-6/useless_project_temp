#!/usr/bin/env python3

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

POSES = ["time_out", "heart", "cover_nose", "crashing_out", "dance", "nose_closed", "flirty", "hand_up",
         "tongue_out", "open_mouth", "disgusted", "talking_to_wall", "suspicious", "spin"]
TEST_KEYS = "1234567890-=[]"

FACE_SCALE = 2.0
HOLD_FRAMES = 10
ARM = {
    "spin": 15, "suspicious": 8, "talking_to_wall": 6, "dance": 6, "crashing_out": 4,
    "open_mouth": 4, "tongue_out": 5, "disgusted": 5,
}

Z = dict(
    jaw_open=6.0,
    scream_jaw=3.5,
    tongue_jaw=3.5,
    sneer=4.5,
    disgust=14.0,
    squint=4.0,
)
Z_CAP = 8.0
FLOOR = dict(
    jaw_open=0.30,
    scream_jaw=0.18,
    tongue_jaw=0.18,
    sneer=0.06,
    squint=0.18,
)
T = dict(
    tongue=0.5,
    head_turn=0.15,
    gesture=0.035,
)

CALIB_FILE = "calibration.json"
CALIB_SECONDS = 7.0
CALIB_WARMUP = 1.5
CALIB_MIN_SAMPLES = 30
SIGMA_FLOOR = 0.015
SIGMA_CEIL = 0.080
CALIB_VERSION = 1

GENERIC_SIGMA = 0.035
GENERIC_MEAN = {
    "jawOpen": 0.08, "eyeSquintLeft": 0.10, "eyeSquintRight": 0.10,
    "eyeBlinkLeft": 0.10, "eyeBlinkRight": 0.10, "noseSneerLeft": 0.03, "noseSneerRight": 0.03,
    "browDownLeft": 0.06, "browDownRight": 0.06, "mouthFrownLeft": 0.05, "mouthFrownRight": 0.05,
    "mouthUpperUpLeft": 0.05, "mouthUpperUpRight": 0.05,
}

INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
HERE = os.path.dirname(os.path.abspath(__file__))


def ensure_models():
    mdir = os.path.join(HERE, "models")
    os.makedirs(mdir, exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        path = os.path.join(mdir, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def preflight(model_path):
    """Open a detector in a throwaway subprocess: bad macOS builds abort() uncatchably."""
    code = (
        "import sys\n"
        "from mediapipe.tasks import python as t\n"
        "from mediapipe.tasks.python import vision\n"
        "vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(\n"
        "    base_options=t.BaseOptions(model_asset_path=sys.argv[1]),\n"
        "    running_mode=vision.RunningMode.VIDEO, num_faces=1,\n"
        "    output_face_blendshapes=True)).close()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, model_path], capture_output=True, text=True)
    if proc.returncode == 0:
        return
    err = (proc.stderr or "") + (proc.stdout or "")
    print(f"\nMediaPipe cannot start a detector here (python {platform.python_version()}, "
          f"mediapipe {getattr(mp, '__version__', '?')}, exit {proc.returncode}).\n")
    if "Service is unavailable" in err or "MetalHelper" in err or proc.returncode == -6:
        print("Cause: mediapipe 0.10.30+ ships macOS wheels that abort on startup.\n"
              "Fix (Python 3.11 or 3.12) — install the pinned set:\n"
              "  pip install -r requirements.txt\n"
              "If you already installed something newer by hand, force it back:\n"
              '  pip install "mediapipe==0.10.21" "numpy<2" "opencv-python<5" "opencv-contrib-python<5"\n')
    else:
        print(err[-1500:])
    sys.exit(1)


def build_detectors(model_paths):
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["hand_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["pose_landmarker_lite.task"]),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    return face, hand, pose


class Clock:
    """Strictly increasing timestamps for the life of a detector, recalibrations included."""

    def __init__(self):
        self.t0, self.last = time.monotonic(), -1

    def next(self):
        self.last = max(int((time.monotonic() - self.t0) * 1000), self.last + 1)
        return self.last


class Baseline:
    """Your resting face: a mean and a wobble for every channel."""

    def __init__(self, mean=None, sigma=None, samples=0, made=None, learned=None, roasts=None, sounds=None):
        self.mean = mean or {}
        self.sigma = sigma or {}
        self.samples = samples
        self.made = made
        self.learned = learned or {}
        self.roasts = roasts or {}
        self.sounds = sounds or {}
        self.generic = not self.mean

    def z(self, name, value):
        """How far above your neutral this channel is, in standard deviations."""
        if self.generic:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        m = self.mean.get(name)
        if m is None:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        return (value - m) / self.sigma.get(name, SIGMA_CEIL)

    @property
    def neutral_turn(self):
        return self.mean.get("turn_signed", 0.0) if not self.generic else 0.0

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"version": CALIB_VERSION, "made": self.made, "samples": self.samples,
                       "mean": self.mean, "sigma": self.sigma, "learned": self.learned, "roasts": self.roasts, "sounds": getattr(self, "sounds", {})}, fh, indent=1, sort_keys=True)

    @staticmethod
    def load(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return Baseline()
        # Keep learned reactions even when no facial calibration exists yet.
        # This is important because the learned reaction data is stored in the
        # existing calibration.json file.
        if data.get("version") != CALIB_VERSION:
            return Baseline(learned=data.get("learned", {}), roasts=data.get("roasts", {}), sounds=data.get("sounds", {}))
        return Baseline(data.get("mean", {}), data.get("sigma", {}),
                        data.get("samples", 0), data.get("made"),
                        data.get("learned", {}), data.get("roasts", {}), data.get("sounds", {}))


class Collector:
    """Running mean and standard deviation per channel, over the calibration window."""

    def __init__(self):
        self.n, self.s, self.ss = 0, {}, {}

    def add(self, face):
        self.n += 1
        for name, v in list(face.bs.items()) + [("turn_signed", face.turn_signed)]:
            self.s[name] = self.s.get(name, 0.0) + v
            self.ss[name] = self.ss.get(name, 0.0) + v * v

    def finish(self):
        mean, sigma = {}, {}
        for name, total in self.s.items():
            m = total / self.n
            var = max(self.ss[name] / self.n - m * m, 0.0)
            mean[name] = round(m, 5)
            sigma[name] = round(min(max(var ** 0.5, SIGMA_FLOOR), SIGMA_CEIL), 5)
        sigma["turn_signed"] = min(max(sigma.get("turn_signed", 0.02), 0.01), 0.10)
        return Baseline(mean, sigma, self.n, time.strftime("%Y-%m-%d %H:%M"))


def calibration_warnings(base):
    """Catch the two ways a calibration goes wrong: mid-expression, or fidgeting."""
    out = []
    if base.mean.get("jawOpen", 0) > 0.30:
        out.append("your mouth looks like it was open — don't talk during calibration")
    if max(base.mean.get("noseSneerLeft", 0), base.mean.get("noseSneerRight", 0)) > 0.15:
        out.append("your nose was scrunched — hold a bored face, not a reaction")
    if max(base.mean.get("browInnerUp", 0), base.mean.get("browOuterUpLeft", 0)) > 0.35:
        out.append("your eyebrows were up — relax them")
    pinned = sum(1 for k, v in base.sigma.items() if v >= SIGMA_CEIL)
    if pinned > 12:
        out.append("you moved a lot — sit still and try again for a tighter baseline")
    return out


def run_calibration(cap, face_det, clock, args, W, H, window):
    """Watch a bored face for CALIB_SECONDS and learn what its channels rest at."""
    print(f"\nCalibrating for {CALIB_SECONDS:.0f}s. Sit how you normally sit, look at the "
          "camera, hold a bored face.\nBlinking is fine. Don't talk, smile or raise your eyebrows.")
    col, start, seen_face = Collector(), time.monotonic(), 0
    while True:
        elapsed = time.monotonic() - start
        if elapsed > CALIB_SECONDS:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != H or frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H))
        if not args.no_flip:
            frame = cv2.flip(frame, 1)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fr = face_det.detect_for_video(mp_img, clock.next())
        face = Face(fr.face_landmarks[0], fr.face_blendshapes[0] if fr.face_blendshapes else None, W, H) \
            if fr.face_landmarks else None
        if face is not None:
            seen_face += 1
            if elapsed > CALIB_WARMUP and face.bs:
                col.add(face)
        draw_calibration(frame, elapsed, col.n, face)
        cv2.imshow(window, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            print("Calibration cancelled.")
            return None

    if col.n < CALIB_MIN_SAMPLES:
        print(f"Calibration failed: only {col.n} usable frames"
              f"{' — your face was never detected' if not seen_face else ''}.\n"
              "  - light your face from the front, sit head-and-shoulders in frame, and try again")
        return None
    base = col.finish()
    print(f"Calibrated on {base.samples} frames. Your neutral face:")
    for name in ("jawOpen", "noseSneerLeft", "browDownLeft", "mouthFrownLeft", "eyeSquintLeft"):
        if name in base.mean:
            print(f"  {name:16s} {base.mean[name]:.3f} ± {base.sigma[name]:.3f}")
    for w in calibration_warnings(base):
        print(f"  ! {w}")
    return base


def draw_calibration(img, elapsed, samples, face):
    H, W = img.shape[:2]
    left = max(0.0, CALIB_SECONDS - elapsed)
    cv2.rectangle(img, (0, 0), (W, 96), (0, 0, 0), -1)
    cv2.putText(img, "CALIBRATING - hold a bored face", (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(img, f"{left:0.1f}s   {samples} frames" + ("" if face is not None else "   NO FACE"),
                (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255) if face is not None else (0, 140, 255), 2)
    done = int(W * min(elapsed / CALIB_SECONDS, 1.0))
    cv2.rectangle(img, (0, 84), (done, 96), (0, 220, 0), -1)
    if face is not None:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 220, 0), 1)


class Asset:
    """One reaction: a list of BGRA frames plus per-frame durations (ms) for GIFs."""

    def __init__(self, frames, durations):
        self.frames = frames
        self.durations = durations
        self.cum = np.cumsum(durations)
        self.total = int(self.cum[-1])
        h, w = frames[0].shape[:2]
        self.aspect = w / h
        self._cache = {}

    def frame_at(self, ms):
        if len(self.frames) == 1:
            return 0
        return int(np.searchsorted(self.cum, ms % self.total, side="right"))

    def scaled(self, idx, height):
        key = (idx, height)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            w = max(1, int(round(height * self.aspect)))
            self._cache[key] = cv2.resize(self.frames[idx], (w, height), interpolation=cv2.INTER_AREA)
        return self._cache[key]


def to_bgra(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def placeholder(label):
    img = np.zeros((300, 300, 4), np.uint8)
    cv2.circle(img, (150, 150), 140, (0, 0, 255, 220), -1)
    cv2.putText(img, label, (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)
    return Asset([img], [100])


def find_asset_file(pose):
    adir = os.path.join(HERE, "assets")
    if not os.path.isdir(adir):
        return None
    exts = (".gif", ".png", ".jpg", ".jpeg")
    for fn in sorted(os.listdir(adir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in exts and (stem == pose or stem.endswith("_" + pose)):
            return os.path.join(adir, fn)
    return None


def load_asset(pose):
    path = find_asset_file(pose)
    if path is None:
        print(f"  {pose:16s} missing -> placeholder")
        return placeholder(pose)
    frames, durations = [], []
    if path.lower().endswith(".gif"):
        from PIL import Image, ImageSequence
        with Image.open(path) as im:
            for f in ImageSequence.Iterator(im):
                frames.append(cv2.cvtColor(np.array(f.convert("RGBA")), cv2.COLOR_RGBA2BGRA))
                durations.append(max(20, int(f.info.get("duration", 100))))
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            frames, durations = [to_bgra(img)], [100]
    if not frames:
        print(f"  {pose:16s} could not read {os.path.basename(path)} -> placeholder")
        return placeholder(pose)
    print(f"  {pose:16s} {os.path.basename(path)}  ({len(frames)} frame{'s' if len(frames) > 1 else ''})")
    return Asset(frames, durations)


SOUND_EXTS = (".wav", ".mp3", ".ogg")


def find_sound_file(label):
    sdir = os.path.join(HERE, "sounds")
    if not os.path.isdir(sdir):
        return None
    for fn in sorted(os.listdir(sdir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in SOUND_EXTS and (stem == label or stem.endswith("_" + label)):
            return os.path.join(sdir, fn)
    return None


def play_sound(path):
    """Play a reaction sound without blocking the camera loop. WAV works on Windows without extra packages."""
    if not path or not os.path.isfile(path):
        return
    try:
        if platform.system().lower().startswith("win") and path.lower().endswith(".wav"):
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
            return
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
        except Exception as e:
            print(f"Could not play sound '{os.path.basename(path)}': {e}")
    except Exception as e:
        print(f"Could not play sound '{os.path.basename(path)}': {e}")


def sound_for(label, custom_sounds):
    configured = custom_sounds.get(label)
    if configured:
        path = configured if os.path.isabs(configured) else os.path.join(HERE, "sounds", configured)
        if os.path.isfile(path):
            return path
    return find_sound_file(label)


def draw_sound_editor(img, labels, index, state, edit_name, edit_text, custom_sounds):
    """S-key UI: list every reaction and assign/preview its sound."""
    H, W = img.shape[:2]
    shade = img.copy()
    cv2.rectangle(shade, (0, 0), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.88, img, 0.12, 0, img)
    white, gray = (255, 255, 255), (190, 190, 190)
    cv2.putText(img, "IT'S GIVING - SOUND EDITOR", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.95, white, 2, cv2.LINE_AA)
    if not labels:
        cv2.putText(img, "No reactions available.", (35, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, gray, 1, cv2.LINE_AA)
        cv2.putText(img, "ESC = close", (35, H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA)
        return
    if state == "list":
        cv2.putText(img, "Select a reaction to view, preview or edit its sound", (35, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, gray, 1, cv2.LINE_AA)
        visible = max(1, min(10, (H - 180) // 48))
        start = max(0, min(index - visible + 1, len(labels) - visible))
        for row, label in enumerate(labels[start:start + visible]):
            y = 140 + row * 48
            selected = (start + row) == index
            if selected:
                cv2.rectangle(img, (25, y - 30), (W - 25, y + 12), (45, 45, 45), -1)
            path = sound_for(label, custom_sounds)
            sound_name = os.path.basename(path) if path else "(no sound)"
            cv2.putText(img, ("> " if selected else "  ") + label, (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, white, 1, cv2.LINE_AA)
            cv2.putText(img, sound_name[:35], (W - 360, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, gray, 1, cv2.LINE_AA)
        cv2.putText(img, "ENTER = edit   P = preview   UP/DOWN = select   ESC = close", (35, H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, f"Editing sound for: {edit_name}", (35, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.72, white, 2, cv2.LINE_AA)
        cv2.rectangle(img, (35, 140), (W - 35, 225), (35, 35, 35), -1)
        cv2.putText(img, edit_text or "_", (55, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 1, cv2.LINE_AA)
        cv2.putText(img, "Type a filename inside sounds/ (e.g. shocked.wav)", (35, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.58, gray, 1, cv2.LINE_AA)
        cv2.putText(img, "Leave blank to use an automatically matched file.", (35, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.58, gray, 1, cv2.LINE_AA)
        cv2.putText(img, "ENTER = save   BACKSPACE = delete   ESC = cancel", (35, H - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA)


ROASTS = {
    "time_out": "Timeout! Even your brain needs a break from you.",
    "heart": "Aww, a heart. Someone is dangerously wholesome today.",
    "cover_nose": "That smell must be your own bad decisions.",
    "crashing_out": "The last brain cell has officially logged off.",
    "dance": "Okay dancer, nobody asked for the full performance package.",
    "nose_closed": "Nose closed. Apparently common sense is closed too.",
    "flirty": "Ohhh, suddenly we have a main-character moment.",
    "hand_up": "Hand up! Please wait while your confidence loads.",
    "tongue_out": "Tongue out? Professionalism has left the chat.",
    "open_mouth": "Close your mouth. The Wi-Fi isn't coming through it.",
    "disgusted": "That face says: absolutely not, delete the whole situation.",
    "talking_to_wall": "Talking to the wall? Finally, someone who understands you.",
    "suspicious": "That side-eye is LOUD. Somebody is under investigation.",
    "spin": "Spinning like the plot still makes sense. It doesn't.",
}


def auto_roast(label):
    """Playful fallback roast for any newly learned reaction."""
    name = str(label).replace("_", " ").strip() or "that reaction"
    return f"{name.title()} detected. Bro really thought this was a personality trait."


def roast_for(label, custom_roasts):
    return custom_roasts.get(label) or ROASTS.get(label) or auto_roast(label)


def draw_roast_editor(img, labels, index, state, edit_name, edit_text):
    """E-key UI: list every reaction and edit its roast."""
    H, W = img.shape[:2]
    shade = img.copy()
    cv2.rectangle(shade, (0, 0), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.88, img, 0.12, 0, img)
    white, gray = (255, 255, 255), (190, 190, 190)
    cv2.putText(img, "IT'S GIVING - ROAST EDITOR", (35, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, white, 2, cv2.LINE_AA)
    if not labels:
        cv2.putText(img, "No reactions available.", (35, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, gray, 1, cv2.LINE_AA)
        cv2.putText(img, "ESC = close", (35, H - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA)
        return
    if state == "list":
        cv2.putText(img, "Select a pose/reaction to view or edit its roast", (35, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, gray, 1, cv2.LINE_AA)
        visible = max(1, min(10, (H - 170) // 48))
        start = max(0, min(index - visible + 1, len(labels) - visible))
        for row, label in enumerate(labels[start:start + visible]):
            y = 140 + row * 48
            selected = (start + row) == index
            if selected:
                cv2.rectangle(img, (25, y - 30), (W - 25, y + 12), (45, 45, 45), -1)
            cv2.putText(img, ("> " if selected else "  ") + label, (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, white, 1, cv2.LINE_AA)
        current = roast_for(labels[index], {}) if labels else ""
        cv2.putText(img, "Current: " + current[:85], (35, H - 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, gray, 1, cv2.LINE_AA)
        cv2.putText(img, "UP/DOWN = select   ENTER = edit   ESC = close", (35, H - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, f"Editing roast for: {edit_name}", (35, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, white, 2, cv2.LINE_AA)
        cv2.rectangle(img, (35, 140), (W - 35, 250), (35, 35, 35), -1)
        words = edit_text.split()
        lines, line = [], ""
        for word in words:
            test = (line + " " + word).strip()
            if line and cv2.getTextSize(test, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 1)[0][0] > W - 100:
                lines.append(line)
                line = word
            else:
                line = test
        if line:
            lines.append(line)
        for i, line in enumerate(lines[:3]):
            cv2.putText(img, line, (55, 180 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, white, 1, cv2.LINE_AA)
        cv2.putText(img, "Type your custom roast", (35, 290),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, gray, 1, cv2.LINE_AA)
        cv2.putText(img, "ENTER = save   BACKSPACE = delete   ESC = cancel", (35, H - 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1, cv2.LINE_AA)


def draw_roast(frame, text, x, y, width):
    """Draw a black caption box with wrapped white roast text below the reaction."""
    if not text:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 1
    max_text_width = max(180, min(width, 620) - 24)
    words = text.split()
    lines, line = [], ""
    for word in words:
        test = (line + " " + word).strip()
        tw = cv2.getTextSize(test, font, scale, thickness)[0][0]
        if line and tw > max_text_width:
            lines.append(line)
            line = word
        else:
            line = test
    if line:
        lines.append(line)

    line_h = 25
    box_w = min(max_text_width + 24, max(180, width))
    box_h = line_h * len(lines) + 18
    H, W = frame.shape[:2]
    bx = max(5, min(int(x), W - int(box_w) - 5))
    by = int(y)
    if by + box_h > H - 5:
        by = max(5, by - box_h - 12)
    cv2.rectangle(frame, (bx, by), (bx + int(box_w), by + box_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (bx + 12, by + 20 + i * line_h),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def overlay(frame, sprite, x, y):
    """Alpha-composite BGRA sprite onto BGR frame at top-left (x, y), clipped to the frame."""
    H, W = frame.shape[:2]
    h, w = sprite.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return frame
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (a * s[:, :, :3] + (1 - a) * roi).astype(np.uint8)
    return frame


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class Face:
    def __init__(self, lms, blendshapes, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.pts = p
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        self.box = (int(x0), int(y0), int(x1), int(y1))
        self.w, self.h = float(x1 - x0), float(y1 - y0)
        self.center = ((x0 + x1) / 2, (y0 + y1) / 2)
        self.nose, self.chin, self.top = p[1], p[152], p[10]
        self.mouth = (p[13] + p[14]) / 2
        self.eye_y = float((p[33][1] + p[263][1]) / 2)
        cl, cr = p[234], p[454]
        self.turn_signed = float((self.nose[0] - cl[0]) / max(cr[0] - cl[0], 1e-3) - 0.5)
        self.bs = {c.category_name: c.score for c in (blendshapes or [])}

    def b(self, name):
        return self.bs.get(name, 0.0)


HAND_CONNECTIONS = ((0,1),(1,2),(2,3),(3,4),
                    (0,5),(5,6),(6,7),(7,8),
                    (0,9),(9,10),(10,11),(11,12),
                    (0,13),(13,14),(14,15),(15,16),
                    (0,17),(17,18),(18,19),(19,20),
                    (5,9),(9,13),(13,17))

class Hand:
    def __init__(self, lms, W, H, handedness=None):
        # Keep all 21 MediaPipe landmarks.  The old code only used 3 fingertips,
        # which made very different hand shapes look almost identical.
        self.p = np.array([[l.x * W, l.y * H, getattr(l, "z", 0.0) * W]
                           for l in lms], np.float32)
        p = self.p[:, :2]
        self.handedness = str(handedness or "unknown").lower()
        self.wrist = p[0]
        self.palm = p[[0, 5, 9, 13, 17]].mean(0)
        self.thumb, self.index, self.middle = p[4], p[8], p[12]
        self.ring, self.pinky = p[16], p[20]
        d = p[9] - p[0]
        self.horizontal = abs(d[0]) > 1.5 * abs(d[1])
        self.vertical = abs(d[1]) > 1.5 * abs(d[0])
        self.scale = max(dist(p[0], p[9]), dist(p[5], p[17]), 1.0)
        # Finger extension is measured against the complete finger geometry, not
        # only fingertip-vs-middle-joint distance.
        self.extended = self._extended_fingers()
        self.open = sum(self.extended) >= 3

    def _extended_fingers(self):
        p = self.p
        # Thumb uses a different geometry; the other four fingers use tip distance
        # plus joint direction to make curled/extended fingers distinguishable.
        thumb = dist(p[4,:2], p[5,:2]) > 0.95 * dist(p[3,:2], p[5,:2])
        fingers = []
        for m, pip, dip, tip in ((5,6,7,8),(9,10,11,12),(13,14,15,16),(17,18,19,20)):
            straight = (dist(p[m,:2], p[tip,:2]) > 1.45 * dist(p[m,:2], p[pip,:2])
                        and dist(p[pip,:2], p[tip,:2]) > 1.20 * dist(p[pip,:2], p[dip,:2]))
            fingers.append(bool(straight))
        return [bool(thumb)] + fingers


class Body:
    """Upper-body pose: shoulders 11/12, elbows 13/14, wrists 15/16."""

    def __init__(self, lms, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.shoulders, self.elbows, self.wrists = p[[11, 12]], p[[13, 14]], p[[15, 16]]
        vis = [getattr(lms[i], "visibility", 1.0) for i in (11, 12, 13, 14)]
        self.seen = min(vis) > 0.5
        shoulder_y = float(self.shoulders[:, 1].mean())
        self.elbows_up = self.seen and bool((self.elbows[:, 1] < shoulder_y).all())


def tongue_score(frame, face, hands, jaw_ready):
    """Fraction of the mouth opening that reads pink: saturated and lit, unlike teeth or throat."""
    if not jaw_ready:
        return 0.0
    if any(dist(h.palm, face.mouth) < 0.7 * face.w for h in hands):
        return 0.0
    poly = face.pts[INNER_LIPS].astype(np.int32)
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    x0, y0 = max(x0, 0), max(y0, 0)
    roi = frame[y0:y1 + 1, x0:x1 + 1]
    if roi.size == 0:
        return 0.0
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - [x0, y0]], 255)
    k = max(3, int(0.15 * (y1 - y0)))
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    n = int(np.count_nonzero(mask))
    if n < 40:
        return 0.0
    h, s, v = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    pink = ((h < 12) | (h > 160)) & (s > 70) & (v > 110)
    return float(np.count_nonzero(pink & (mask > 0)) / n)


class Motion:
    """Smoothed hand speed across frames, in face-widths per frame."""

    def __init__(self):
        self.prev, self.energy, self.fw = [], 0.0, 200.0

    def update(self, hands, face):
        if face is not None:
            self.fw = max(face.w, 1.0)
        cur = [h.palm for h in hands]
        speed = 0.0
        if cur and self.prev:
            moved = [min(dist(c, p) for p in self.prev) for c in cur]
            moved = [m for m in moved if m < self.fw]
            if moved:
                speed = max(moved) / self.fw
        self.energy = 0.8 * self.energy + 0.2 * speed
        self.prev = cur
        return self.energy


def measure(face, base):
    """Every expression channel, raw and in sigma above your own neutral."""
    zpair = lambda n: (base.z(n + "Left", face.b(n + "Left")) + base.z(n + "Right", face.b(n + "Right"))) / 2
    pair = lambda n: (face.b(n + "Left") + face.b(n + "Right")) / 2
    m = {
        "jaw": face.b("jawOpen"), "z_jaw": base.z("jawOpen", face.b("jawOpen")),
        "sneer": pair("noseSneer"), "z_sneer": zpair("noseSneer"),
        "z_brow": zpair("browDown"), "z_frown": zpair("mouthFrown"), "z_lip": zpair("mouthUpperUp"),
        "squint": max(pair("eyeSquint"), pair("eyeBlink")),
        "z_squint": max(zpair("eyeSquint"), zpair("eyeBlink")),
        "turn": abs(face.turn_signed - base.neutral_turn),
    }
    cap = lambda v: min(v, Z_CAP)
    m["z_disgust"] = 2 * cap(m["z_sneer"]) + cap(m["z_brow"]) + cap(m["z_frown"]) + cap(m["z_lip"])
    return m


def over(key, m, zkey, rawkey):
    """Sigma above your neutral AND a raw floor, so a tiny sigma can't become a hair trigger."""
    return m[zkey] >= Z[key] and m[rawkey] >= FLOOR[key]


def decide(face, hands, body, tongue, gesture, m):
    """Return (pose or None, debug dict)."""
    d = {"hands": len(hands)}
    if face is None:
        gone = not hands and (body is None or not body.seen)
        return ("spin" if gone else None), d

    fw = face.w
    near = lambda a, b, k: dist(a, b) < k * fw
    elbows_up = bool(body and body.elbows_up)
    d.update(m, gesture=gesture, tongue=tongue, elbows_up=elbows_up)
    screaming = over("scream_jaw", m, "z_jaw", "jaw")

    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        for top, under in ((a, b), (b, a)):
            if top.horizontal and under.vertical and top.palm[1] < under.palm[1] \
                    and near(under.middle, top.palm, 0.6):
                return "time_out", d
        if near(a.index, b.index, 0.3) and near(a.thumb, b.thumb, 0.3) \
                and (a.index[1] + b.index[1]) < (a.thumb[1] + b.thumb[1]):
            return "heart", d
        if near(a.palm, face.mouth, 0.6) and near(b.palm, face.mouth, 0.6):
            return "cover_nose", d
        on_head = lambda h: (h.palm[1] < face.eye_y and abs(h.palm[0] - face.nose[0]) < 1.1 * fw
                             and h.palm[1] > face.top[1] - 0.8 * face.h)
        if on_head(a) and on_head(b) and screaming:
            return "crashing_out", d

    near_head = lambda h: abs(h.palm[0] - face.nose[0]) < 1.3 * fw and h.palm[1] < face.eye_y + 0.3 * face.h
    if elbows_up and all(near_head(h) for h in hands):
        return ("crashing_out" if screaming else "dance"), d

    for h in hands:
        if near(h.thumb, face.nose, 0.35) and near(h.index, face.nose, 0.35) and near(h.thumb, h.index, 0.3):
            return "nose_closed", d
        if near(h.index, face.mouth, 0.22) and not near(h.palm, face.mouth, 0.3):
            return "flirty", d
        if h.open and h.palm[1] < face.nose[1] and abs(h.palm[0] - face.nose[0]) > 0.8 * fw:
            return "hand_up", d

    if tongue > T["tongue"]:
        return "tongue_out", d
    if over("jaw_open", m, "z_jaw", "jaw"):
        return "open_mouth", d
    if over("sneer", m, "z_sneer", "sneer") or m["z_disgust"] >= Z["disgust"]:
        return "disgusted", d
    if hands and gesture > T["gesture"]:
        return "talking_to_wall", d
    if m["turn"] > T["head_turn"] and over("squint", m, "z_squint", "squint"):
        return "suspicious", d
    return None, d


def draw_hud(img, shown, raw, d, face, hands, body, base):
    if face:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
    for h in hands:
        cv2.circle(img, (int(h.palm[0]), int(h.palm[1])), 6, (0, 200, 255), -1)
    if body and body.seen:
        for pt in np.vstack([body.shoulders, body.elbows]):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 6, (255, 120, 0), -1)
    g = d.get
    lines = [
        (f"showing: {shown or '-'}   raw: {raw or '-'}   hands: {g('hands', 0)}"
         f"   elbows up: {'Y' if g('elbows_up') else 'n'}", (0, 255, 0)),
        (f"jaw {g('jaw', 0):.2f} = {g('z_jaw', 0):+.1f}s/{Z['jaw_open']:.0f}   "
         f"squint {g('squint', 0):.2f} = {g('z_squint', 0):+.1f}s/{Z['squint']:.0f}   "
         f"tongue {g('tongue', 0):.2f}   turn {g('turn', 0):.2f}   gesture {g('gesture', 0):.3f}", (0, 255, 0)),
        (f"disgust {g('z_disgust', 0):+.1f}s/{Z['disgust']:.0f} = 2x sneer {g('z_sneer', 0):+.1f} "
         f"+ brow {g('z_brow', 0):+.1f} + frown {g('z_frown', 0):+.1f} + lip {g('z_lip', 0):+.1f}", (0, 255, 0)),
        (("NOT CALIBRATED - generic baseline, everything is harder to trigger. press 'c'"
          if base.generic else
          f"calibrated {base.made} on {base.samples} frames   (s = sigma above your neutral)"),
         (0, 140, 255) if base.generic else (200, 200, 200)),
        ("keys: q quit  d hud  c recalibrate  1-9 0 - = [ ] test poses", (0, 255, 0)),
    ]
    for i, (t, colour) in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)



FACE_FEATURE_NAMES = [
    "jawOpen", "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "noseSneerLeft", "noseSneerRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthPucker", "mouthFunnel", "mouthSmileLeft", "mouthSmileRight",
    "mouthLeft", "mouthRight", "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper", "mouthUpperUpLeft", "mouthUpperUpRight",
    "mouthLowerDownLeft", "mouthLowerDownRight", "cheekPuff", "cheekSquintLeft",
    "cheekSquintRight", "eyeWideLeft", "eyeWideRight", "eyeLookDownLeft",
    "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight", "eyeLookOutLeft",
    "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "mouthStretchLeft",
    "mouthStretchRight", "mouthPressLeft", "mouthPressRight", "mouthDimpleLeft",
    "mouthDimpleRight", "mouthClose", "mouthFrownLeft", "mouthFrownRight",
]

def _hand_signature(hand):
    """Detailed normalized 21-point hand skeleton + finger state."""
    if hand is None:
        return np.zeros(21 * 3 + 5, dtype=np.float32)
    p = hand.p
    scale = max(hand.scale, 1.0)
    wrist = p[0]
    vals = []
    for i in range(21):
        vals.extend([(p[i,0] - wrist[0]) / scale,
                     (p[i,1] - wrist[1]) / scale,
                     p[i,2] / scale])
    vals.extend([float(x) for x in hand.extended])
    return np.asarray(vals, dtype=np.float32)

def _ordered_hands(hands):
    """Return stable left/right slots instead of detector-order slots."""
    left = right = None
    unknown = []
    for h in hands:
        if h.handedness.startswith("left") and left is None:
            left = h
        elif h.handedness.startswith("right") and right is None:
            right = h
        else:
            unknown.append(h)
    for h in unknown:
        if left is None:
            left = h
        elif right is None:
            right = h
    return left, right

def feature_vector(face, hands, body, tongue, gesture, m):
    """High-detail multimodal feature vector for learned reactions.

    Face: MediaPipe blendshape channels.
    Hands: all 21 landmarks for BOTH hands, normalized to each palm.
    This is much more discriminative than the previous 12-value vector.
    """
    vals = []
    if face is None:
        vals.extend([0.0] * len(FACE_FEATURE_NAMES))
    else:
        vals.extend([float(face.b(n)) for n in FACE_FEATURE_NAMES])

    left, right = _ordered_hands(hands)
    vals.extend(_hand_signature(left))
    vals.extend(_hand_signature(right))

    # Relative hand-to-face positions help distinguish gestures such as covering
    # the mouth, touching the head, heart hands, and hands raised beside the face.
    fw = max(face.w if face is not None else 200.0, 1.0)
    face_center = face.center if face is not None else np.array([0.0, 0.0])
    for h in (left, right):
        if h is None:
            vals.extend([0.0, 0.0, 0.0, 0.0])
        else:
            vals.extend([(h.palm[0] - face_center[0]) / fw,
                         (h.palm[1] - face_center[1]) / fw,
                         (h.index[0] - face_center[0]) / fw,
                         (h.index[1] - face_center[1]) / fw])

    vals.extend([float(tongue), float(gesture),
                 float(bool(body and body.elbows_up))])
    return np.asarray(vals, dtype=np.float32)

def predict_learned(learned, vec):
    """Nearest learned reaction using the detailed multimodal skeleton."""
    if not learned:
        return None, float("inf")
    best_label, best_dist = None, float("inf")
    for label, info in learned.items():
        center = np.asarray(info.get("center", []), dtype=np.float32)
        scale = np.asarray(info.get("scale", []), dtype=np.float32)
        if center.shape != vec.shape:
            continue
        if scale.shape != vec.shape:
            # Backward compatibility with older learned reactions.
            scale = np.ones_like(vec, dtype=np.float32)
            scale[:len(FACE_FEATURE_NAMES)] = 0.20
            scale[len(FACE_FEATURE_NAMES):] = 0.35
        scale = np.maximum(scale, 0.04)
        d = float(np.sqrt(np.mean(((vec - center) / scale) ** 2)))
        if d < best_dist:
            best_label, best_dist = label, d
    return (best_label, best_dist) if best_dist <= 1.65 else (None, best_dist)


def draw_delete_ui(img, labels, selected):
    """Show learned reactions in a numbered list for deletion."""
    H, W = img.shape[:2]
    shade = img.copy()
    cv2.rectangle(shade, (0, 0), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(shade, 0.82, img, 0.18, 0, img)
    white, gray = (255, 255, 255), (190, 190, 190)
    cv2.putText(img, "IT'S GIVING - DELETE REACTION", (40, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, white, 2, cv2.LINE_AA)
    cv2.putText(img, "Select a learned reaction to remove", (40, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, gray, 1, cv2.LINE_AA)
    if not labels:
        cv2.putText(img, "No learned reactions yet.", (40, 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, white, 2, cv2.LINE_AA)
    else:
        y = 165
        for i, label in enumerate(labels):
            active = i == selected
            if active:
                cv2.rectangle(img, (30, y - 30), (W - 30, y + 10), (45, 45, 45), -1)
            text = f"{i + 1}. {label}"
            cv2.putText(img, text, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 0.72, white, 2 if active else 1, cv2.LINE_AA)
            y += 48
            if y > H - 100:
                break
        cv2.putText(img, "UP/DOWN = select   ENTER = delete   ESC = cancel", (40, H - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, white, 1, cv2.LINE_AA)


def draw_learn_ui(img, state, name, count, total):
    """Full-screen-ish learning overlay drawn inside the existing OpenCV window."""
    H, W = img.shape[:2]
    overlay_img = img.copy()
    cv2.rectangle(overlay_img, (0, 0), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay_img, 0.82, img, 0.18, 0, img)
    white = (255, 255, 255)
    gray = (190, 190, 190)
    green = (80, 230, 120)
    cv2.putText(img, "IT'S GIVING - LEARN NEW REACTION", (40, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, white, 2, cv2.LINE_AA)
    if state == "name":
        cv2.putText(img, "Type the reaction name and press ENTER", (40, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, gray, 2, cv2.LINE_AA)
        shown = name if name else "_"
        cv2.rectangle(img, (40, 175), (W - 40, 245), (35, 35, 35), -1)
        cv2.putText(img, shown, (60, 222), cv2.FONT_HERSHEY_SIMPLEX, 1.0, white, 2, cv2.LINE_AA)
        cv2.putText(img, "Example: shocked", (40, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.65, gray, 1, cv2.LINE_AA)
        cv2.putText(img, "The matching GIF/photo must already be inside assets/", (40, 335),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, gray, 1, cv2.LINE_AA)
        cv2.putText(img, "ENTER = start    BACKSPACE = edit    ESC = cancel", (40, H - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, white, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, f"Learning: {name}", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.82, white, 2, cv2.LINE_AA)
        cv2.putText(img, "Perform the expression/pose and HOLD it", (40, 185),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, gray, 1, cv2.LINE_AA)
        frac = min(count / max(total, 1), 1.0)
        bar_x, bar_y, bar_w, bar_h = 40, 245, W - 80, 38
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (45, 45, 45), -1)
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_w * frac), bar_y + bar_h), green, -1)
        cv2.putText(img, f"{count}/{total} samples", (40, 325), cv2.FONT_HERSHEY_SIMPLEX, 0.72, white, 2, cv2.LINE_AA)
        cv2.putText(img, "Keep your face/body visible. ESC = cancel", (40, H - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, white, 1, cv2.LINE_AA)

def draw_detection_skeleton(frame, face, hands, body, show=True):
    """Draw the landmarks used by the detector so you can see what it understands."""
    if not show:
        return
    if face is not None:
        # Lightweight face landmarks: eyes, brows, nose and mouth outline.
        face_groups = ((33,133,159,145,153,144,163,7),
                       (263,362,386,374,380,373,390,249),
                       (61,146,91,181,84,17,314,405,321,375,291))
        for group in face_groups:
            pts = [tuple(np.int32(face.pts[i])) for i in group]
            for a, b in zip(pts, pts[1:] + pts[:1]):
                cv2.line(frame, a, b, (80, 220, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, tuple(np.int32(face.nose)), 3, (255, 255, 255), -1)

    for hand in hands:
        for a, b in HAND_CONNECTIONS:
            p1 = tuple(np.int32(hand.p[a, :2]))
            p2 = tuple(np.int32(hand.p[b, :2]))
            cv2.line(frame, p1, p2, (120, 255, 120), 2, cv2.LINE_AA)
        for i, pt in enumerate(hand.p):
            cv2.circle(frame, tuple(np.int32(pt[:2])), 3 if i not in (4,8,12,16,20) else 5,
                       (255, 255, 255), -1, cv2.LINE_AA)
        label = hand.handedness.title()
        cv2.putText(frame, label, tuple(np.int32(hand.wrist - [0, 12])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if body is not None and body.seen:
        pts = np.vstack((body.shoulders, body.elbows, body.wrists)).astype(np.int32)
        cv2.line(frame, tuple(pts[0]), tuple(pts[2]), (255, 180, 80), 2, cv2.LINE_AA)
        cv2.line(frame, tuple(pts[1]), tuple(pts[3]), (255, 180, 80), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), 4, (255, 255, 255), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is your iPhone)")
    ap.add_argument("--calibrate", action="store_true", help="learn your neutral face, save it, and exit")
    ap.add_argument("--no-calibration", action="store_true", help="ignore calibration.json; use the generic baseline")
    ap.add_argument("--no-vcam", action="store_true", help="preview only; don't start the virtual camera")
    ap.add_argument("--skip-check", action="store_true", help="skip the MediaPipe startup check")
    ap.add_argument("--size", default="1280x720", help="capture size, e.g. 1280x720 or 640x480 (lower = faster)")
    ap.add_argument("--no-flip", action="store_true", help="don't mirror the image")
    args = ap.parse_args()

    calib_path = os.path.join(HERE, CALIB_FILE)
    base = Baseline() if args.no_calibration else Baseline.load(calib_path)
    if base.generic and not args.calibrate and not args.no_calibration:
        print(f"No usable {CALIB_FILE}. Running on the generic baseline — everything is harder to\n"
              f"trigger than it should be. Run:  python {os.path.basename(__file__)} --calibrate")

    model_paths = ensure_models()
    if not args.skip_check:
        preflight(model_paths["face_landmarker.task"])

    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened() and "x" in args.size:
        w, h = args.size.lower().split("x")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    ok, frame = False, None
    if cap.isOpened():
        for _ in range(5):
            ok, frame = cap.read()
            if not ok:
                break
    if not ok:
        sys.exit(f"Could not read from camera {args.camera}.\n"
                 "  - try --camera 1\n"
                 "  - System Settings > Privacy & Security > Camera: allow your terminal app, then re-run")
    H, W = frame.shape[:2]
    print(f"Camera {args.camera}: {W}x{H}")

    clock = Clock()
    window = "it's giving v2  (q quit, d HUD, c recalibrate, 1-9 0 - = [ ] test)"

    if args.calibrate:
        face_det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
        try:
            new = run_calibration(cap, face_det, clock, args, W, H, window)
        finally:
            face_det.close()
            cap.release()
            cv2.destroyAllWindows()
        if new is None:
            sys.exit(1)
        new.save(calib_path)
        print(f"Saved {CALIB_FILE}. Now run:  python {os.path.basename(__file__)}")
        return

    print("Assets:")
    assets = {pose: load_asset(pose) for pose in POSES}
    learned = dict(base.learned)
    for label in learned:
        if label not in assets:
            assets[label] = load_asset(label)
    learn_state, learn_name, learn_samples = None, "", []
    custom_roasts = dict(getattr(base, "roasts", {}) or {})
    roast_labels, roast_index, roast_state, roast_name, roast_text = [], 0, None, "", ""
    sound_labels, sound_index, sound_state, sound_name, sound_text = [], 0, None, "", ""
    custom_sounds = dict(getattr(base, "sounds", {}) or {})
    delete_labels, delete_index = [], 0
    LEARN_TOTAL = 90
    learn_distance = float("inf")

    # Keep the assets directory hot-reloaded while the program is running.
    # Adding/replacing an asset file does not require restarting the program.
    asset_dir = os.path.join(HERE, "assets")
    asset_signature = None
    last_asset_scan = 0.0

    def refresh_assets(now):
        nonlocal assets, asset_signature, last_asset_scan
        if now - last_asset_scan < 1.0:
            return
        last_asset_scan = now
        try:
            files = tuple(sorted(
                (fn, os.path.getsize(os.path.join(asset_dir, fn)),
                 os.path.getmtime(os.path.join(asset_dir, fn)))
                for fn in os.listdir(asset_dir)
                if os.path.splitext(fn)[1].lower() in (".gif", ".png", ".jpg", ".jpeg")
            )) if os.path.isdir(asset_dir) else ()
        except OSError:
            files = ()
        if files != asset_signature:
            asset_signature = files
            print("Assets changed - reloading...")
            assets = {pose: load_asset(pose) for pose in POSES}
            for label in learned:
                if label not in assets:
                    assets[label] = load_asset(label)

    vcam = None
    if not args.no_vcam:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(width=W, height=H, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"Virtual camera: '{vcam.device}'  <- pick this camera in Zoom / Meet")
        except Exception as e:
            print(f"Virtual camera unavailable ({e}). Preview-only.")

    face_det, hand_det, pose_det = build_detectors(model_paths)
    motion = Motion()
    shown, hold, show_hud = None, 0, True
    arm = {p: 0 for p in POSES}
    shown_since = 0.0
    forced, forced_until = None, 0.0
    sm_center, sm_h = np.array([W / 2, H / 2], np.float32), H * 0.45
    print("Running. q quit, d HUD, l learn, e edit roasts, s edit sounds, x delete learned reaction, c recalibrate, 1-9 0 - = [ ] test")

    try:
        while True:
            now = time.monotonic()
            refresh_assets(now)
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            if frame.shape[0] != H or frame.shape[1] != W:
                frame = cv2.resize(frame, (W, H))
            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            ts = clock.next()
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fr = face_det.detect_for_video(mp_img, ts)
            hr = hand_det.detect_for_video(mp_img, ts)
            pr = pose_det.detect_for_video(mp_img, ts)
            face = Face(fr.face_landmarks[0], fr.face_blendshapes[0] if fr.face_blendshapes else None, W, H) \
                if fr.face_landmarks else None
            hands = []
            for i, h in enumerate(hr.hand_landmarks):
                handed = None
                if i < len(hr.handedness) and hr.handedness[i]:
                    handed = hr.handedness[i][0].category_name
                hands.append(Hand(h, W, H, handed))
            body = Body(pr.pose_landmarks[0], W, H) if pr.pose_landmarks else None

            m = measure(face, base) if face is not None else {}
            tongue = tongue_score(frame, face, hands,
                                  over("tongue_jaw", m, "z_jaw", "jaw")) if face is not None else 0.0
            gesture = motion.update(hands, face)
            raw, dbg = decide(face, hands, body, tongue, gesture, m)
            learned_raw, learn_distance = (None, float("inf"))
            if face is not None and learn_state != "capture":
                learned_raw, learn_distance = predict_learned(
                    learned, feature_vector(face, hands, body, tongue, gesture, m))
                if learned_raw:
                    raw = learned_raw
                    dbg["learned"] = True
                    dbg["learn_dist"] = learn_distance

            # Collect examples while the in-window Learn UI is active.
            if learn_state == "capture" and face is not None:
                learn_samples.append(feature_vector(face, hands, body, tongue, gesture, m))
                if len(learn_samples) >= LEARN_TOTAL:
                    sample_matrix = np.stack(learn_samples)
                    center = np.mean(sample_matrix, axis=0)
                    # Store per-feature variation so expressive features do not
                    # overpower stable ones during nearest-neighbour matching.
                    scale = np.std(sample_matrix, axis=0)
                    scale = np.clip(scale, 0.04, 0.35)
                    learned[learn_name] = {
                        "center": [round(float(x), 6) for x in center],
                        "scale": [round(float(x), 6) for x in scale],
                        "samples": len(learn_samples),
                        "made": time.strftime("%Y-%m-%d %H:%M"),
                    }
                    assets[learn_name] = load_asset(learn_name)
                    if learn_name not in custom_roasts:
                        custom_roasts[learn_name] = auto_roast(learn_name)
                    base.learned = learned
                    base.roasts = custom_roasts
                    base.sounds = custom_sounds
                    base.save(calib_path)
                    print(f"Learned reaction '{learn_name}' from {len(learn_samples)} samples and saved to {CALIB_FILE}.")
                    learn_state, learn_name, learn_samples = None, "", []
                    motion, shown, hold = Motion(), None, 0
                    arm = {p: 0 for p in POSES}

            fired = None
            # Existing built-in reactions use their original per-pose hold
            # thresholds. Learned reactions get their own small counter so a
            # newly learned label can actually trigger its asset.
            all_reactions = list(POSES) + [p for p in learned if p not in POSES]
            for p in all_reactions:
                if p not in arm:
                    arm[p] = 0
                arm[p] = arm[p] + 1 if raw == p else 0
                threshold = ARM.get(p, 3)
                if raw == p and arm[p] >= threshold:
                    fired = p
            now = time.monotonic()
            if forced and now < forced_until:
                fired = forced
            if fired:
                if fired != shown:
                    shown_since = now
                    if fired:
                        play_sound(sound_for(fired, custom_sounds))
                shown, hold = fired, HOLD_FRAMES
            elif hold > 0:
                hold -= 1
            else:
                shown = None

            if face is not None:
                sm_center = 0.7 * sm_center + 0.3 * np.array(face.center, np.float32)
                sm_h = 0.7 * sm_h + 0.3 * face.h * FACE_SCALE

            draw_detection_skeleton(frame, face, hands, body, show=show_hud)

            if shown:
                asset = assets[shown]
                idx = asset.frame_at(int((now - shown_since) * 1000))
                h = int(min(sm_h, H * 0.98, (W * 0.98) / asset.aspect)) // 8 * 8
                sprite = asset.scaled(idx, max(h, 8))
                sh, sw = sprite.shape[:2]
                sprite_x = int(sm_center[0] - sw / 2)
                sprite_y = int(sm_center[1] - sh / 2 - 0.05 * sh)
                overlay(frame, sprite, sprite_x, sprite_y)
                roast = roast_for(shown, custom_roasts)
                draw_roast(frame, roast, sprite_x, sprite_y + sh + 10, sw)

            if roast_state:
                current = roast_labels[roast_index] if roast_labels else ""
                current_text = roast_text if roast_state == "edit" else roast_for(current, custom_roasts)
                draw_roast_editor(frame, roast_labels, roast_index, roast_state, roast_name, current_text)
            elif sound_state:
                current = sound_labels[sound_index] if sound_labels else ""
                draw_sound_editor(frame, sound_labels, sound_index, sound_state, sound_name, sound_text, custom_sounds)
            elif learn_state:
                draw_learn_ui(frame, learn_state, learn_name, len(learn_samples), LEARN_TOTAL)
            elif delete_labels:
                draw_delete_ui(frame, delete_labels, delete_index)

            if vcam:
                vcam.send(frame)
                vcam.sleep_until_next_frame()

            preview = frame
            if show_hud:
                preview = frame.copy()
                draw_hud(preview, shown, raw, dbg, face, hands, body, base)
            cv2.imshow(window, preview)
            key_raw = cv2.waitKeyEx(1)
            key = key_raw & 0xFF
            if key == ord("q"):
                break
            if roast_state:
                if key == 27:
                    roast_state, roast_name, roast_text = None, "", ""
                elif roast_state == "list":
                    if key_raw in (2490368, 82):
                        roast_index = (roast_index - 1) % len(roast_labels)
                    elif key_raw in (2621440, 84):
                        roast_index = (roast_index + 1) % len(roast_labels)
                    elif key in (10, 13):
                        roast_name = roast_labels[roast_index]
                        roast_text = custom_roasts.get(roast_name, ROASTS.get(roast_name, auto_roast(roast_name)))
                        roast_state = "edit"
                elif roast_state == "edit":
                    if key in (10, 13):
                        if roast_text.strip():
                            custom_roasts[roast_name] = roast_text.strip()
                            base.roasts = custom_roasts
                            base.save(calib_path)
                            print(f"Saved custom roast for '{roast_name}'.")
                        roast_state = "list"
                    elif key in (8, 127):
                        roast_text = roast_text[:-1]
                    elif 32 <= key <= 126 and len(roast_text) < 180:
                        roast_text += chr(key)
            elif sound_state:
                if key == 27:
                    sound_state, sound_name, sound_text = None, "", ""
                elif sound_state == "list":
                    if key_raw in (2490368, 82):
                        sound_index = (sound_index - 1) % len(sound_labels)
                    elif key_raw in (2621440, 84):
                        sound_index = (sound_index + 1) % len(sound_labels)
                    elif key in (10, 13):
                        sound_name = sound_labels[sound_index]
                        current = custom_sounds.get(sound_name, "")
                        if not current:
                            auto_path = find_sound_file(sound_name)
                            current = os.path.basename(auto_path) if auto_path else ""
                        sound_text = current
                        sound_state = "edit"
                    elif key in (ord("p"), ord("P")):
                        label = sound_labels[sound_index]
                        path = sound_for(label, custom_sounds)
                        if path:
                            play_sound(path)
                            print(f"Playing sound for '{label}': {os.path.basename(path)}")
                        else:
                            print(f"No sound assigned to '{label}'.")
                elif sound_state == "edit":
                    if key in (10, 13):
                        value = sound_text.strip()
                        if value:
                            full = os.path.join(HERE, "sounds", value)
                            if not os.path.isfile(full):
                                print(f"Sound file not found: {full}")
                            else:
                                custom_sounds[sound_name] = value
                                base.sounds = custom_sounds
                                base.save(calib_path)
                                print(f"Saved sound for '{sound_name}': {value}")
                                sound_state = "list"
                        else:
                            custom_sounds.pop(sound_name, None)
                            base.sounds = custom_sounds
                            base.save(calib_path)
                            print(f"Saved sound for '{sound_name}': (automatic match)")
                            sound_state = "list"
                    elif key in (8, 127):
                        sound_text = sound_text[:-1]
                    elif 32 <= key <= 126 and len(sound_text) < 120:
                        sound_text += chr(key)
            elif delete_labels:
                if key == 27:
                    delete_labels, delete_index = [], 0
                elif key_raw in (2490368, 82):  # Up arrow (platform-dependent OpenCV code)
                    delete_index = (delete_index - 1) % len(delete_labels)
                elif key_raw in (2621440, 84):  # Down arrow (platform-dependent OpenCV code)
                    delete_index = (delete_index + 1) % len(delete_labels)
                elif key in (10, 13):
                    label = delete_labels[delete_index]
                    if label in learned:
                        del learned[label]
                        base.learned = learned
                        base.save(calib_path)
                    assets.pop(label, None)
                    print(f"Deleted learned reaction '{label}'. The asset file was kept in assets/ so it can be relearned later.")
                    delete_labels = [x for x in delete_labels if x != label]
                    if delete_labels:
                        delete_index = min(delete_index, len(delete_labels) - 1)
                    else:
                        delete_index = 0
                elif 49 <= key <= 57:
                    chosen = key - 49
                    if chosen < len(delete_labels):
                        label = delete_labels[chosen]
                        del learned[label]
                        base.learned = learned
                        base.save(calib_path)
                        assets.pop(label, None)
                        print(f"Deleted learned reaction '{label}'. The asset file was kept in assets/ so it can be relearned later.")
                        delete_labels.pop(chosen)
                        delete_index = min(delete_index, max(0, len(delete_labels) - 1))
            elif learn_state == "name":
                if key == 27:
                    learn_state, learn_name = None, ""
                elif key in (8, 127):
                    learn_name = learn_name[:-1]
                elif key in (10, 13):
                    candidate = learn_name.strip().lower().replace(" ", "_")
                    if not candidate:
                        print("Enter a reaction name first.")
                    elif find_asset_file(candidate) is None:
                        print(f"No asset found for '{candidate}'. Put {candidate}.gif/.png/.jpg in assets first.")
                    else:
                        learn_name, learn_samples, learn_state = candidate, [], "capture"
                        print(f"Learning '{candidate}'. Hold the expression for {LEARN_TOTAL} usable samples.")
                elif 32 <= key <= 126 and len(learn_name) < 30:
                    ch = chr(key).lower()
                    if ch.isalnum() or ch in "_-":
                        learn_name += ch
            elif learn_state == "capture":
                if key == 27:
                    print("Learning cancelled.")
                    learn_state, learn_name, learn_samples = None, "", []
            elif key == ord("e"):
                roast_labels = sorted(set(ROASTS) | set(learned) | set(custom_roasts))
                roast_index = 0
                roast_state = "list" if roast_labels else None
                if roast_labels:
                    print("Roast editor: select a pose, press Enter, type your custom roast, then Enter to save.")
                else:
                    print("Roast editor: no reactions available.")
            elif key == ord("x"):
                delete_labels = sorted(learned.keys())
                delete_index = 0
                if not delete_labels:
                    print("Delete mode: no learned reactions to delete.")
                else:
                    print("Delete mode: select a reaction with number/arrow keys and press Enter.")
            elif key == ord("s"):
                os.makedirs(os.path.join(HERE, "sounds"), exist_ok=True)
                sound_labels = sorted(set(POSES) | set(learned) | set(custom_sounds))
                sound_index = 0
                sound_state = "list" if sound_labels else None
                if sound_labels:
                    print("Sound editor: select a reaction, Enter to assign a sound, P to preview.")
                else:
                    print("Sound editor: no reactions available.")
            elif key == ord("d"):
                show_hud = not show_hud
            elif key == ord("l"):
                learn_state, learn_name, learn_samples = "name", "", []
                shown, hold = None, 0
                print("Learn mode: type the exact asset name (without extension) and press Enter.")
            elif key == ord("c"):
                new = run_calibration(cap, face_det, clock, args, W, H, window)
                if new is not None:
                    base = new
                    base.save(calib_path)
                    print(f"Saved {CALIB_FILE}.")
                motion, shown, hold = Motion(), None, 0
                arm = {p: 0 for p in POSES}
            elif 0 < key < 256 and chr(key) in TEST_KEYS:
                forced, forced_until = POSES[TEST_KEYS.index(chr(key))], now + 2.0
    finally:
        cap.release()
        face_det.close()
        hand_det.close()
        pose_det.close()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
