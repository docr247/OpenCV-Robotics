"""
Finger Tracking → 5-Servo Robotic Hand Control (Robust)
Dependencies: opencv-python, mediapipe, pyfirmata2
"""

import cv2
import mediapipe as mp
import pyfirmata2
import math
import time
from collections import deque

# ================== Hardware Setup ==================
PORT = "/dev/cu.usbmodem1301"   # Change if needed (Windows: "COM3", Linux: "/dev/ttyACM0")
board = pyfirmata2.Arduino(PORT)

# Logical finger order must match tracking: [Thumb, Index, Middle, Ring, Pinky]
# Your wiring: D10 (Thumb), D9 (Index), D6 (Middle), D5 (Ring), D3 (Pinky)
SERVO_PINS = [10, 9, 6, 5, 3]

# Create servo objects in Firmata
servos = [board.get_pin(f"d:{p}:s") for p in SERVO_PINS]

# Explicitly configure each servo's pulse range (standard analog servo)
for p in SERVO_PINS:
    board.servo_config(p, 544, 2400)

# Inversion & per-finger angle limits (tweak as needed)
INVERT = [True, False, False, False, False]  # Thumb inverted only
PER_FINGER_LIMITS = [
    (10, 160),   # Thumb
    (20, 170),   # Index
    (20, 170),   # Middle
    (20, 170),   # Ring
    (20, 170),   # Pinky
]

# ================== Video + MediaPipe Setup ==================
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Landmark indices
TIPS =  [4, 8, 12, 16, 20]
MCPS =  [2, 5, 9, 13, 17]
WRIST = 0
MID_MCP = 9

# ================== Calibration & Smoothing ==================
open_ref = None
closed_ref = None
SMOOTH_N = 5
smoothers = [deque(maxlen=SMOOTH_N) for _ in range(5)]

def dist2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def normalize_metrics(landmarks):
    """Per-finger metric = (tip↔MCP distance) / hand_size; scale-invariant."""
    wrist = landmarks[WRIST]
    mid_mcp = landmarks[MID_MCP]
    hand_size = dist2d(wrist, mid_mcp) or 1e-6
    return [dist2d(landmarks[t], landmarks[m]) / hand_size for t, m in zip(TIPS, MCPS)]

def to_percentages(metrics, open_ref, closed_ref):
    """Map metrics to openness in [0..1] with guards for degenerate calibration."""
    out = []
    for m, o, c in zip(metrics, open_ref, closed_ref):
        if abs(o - c) < 1e-3:
            # If user didn't change that finger during O/C, synthesize a small span
            o = c + 0.05
        out.append(clamp01((m - c) / (o - c)))
    return out

def smooth(values):
    """Tiny moving-average smoothing per finger to avoid jitter."""
    smoothed = []
    for i, v in enumerate(values):
        smoothers[i].append(v)
        smoothed.append(sum(smoothers[i]) / len(smoothers[i]))
    return smoothed

def openness_to_angle(value, amin, amax, invert=False):
    """Map openness [0..1] → servo angle [amin..amax], with optional inversion."""
    value = 1.0 - value if invert else value
    value = clamp01(value)
    return amin + value * (amax - amin)

def sweep(servo, amin, amax, pause=0.25):
    for a in [amin, (amin+amax)//2, amax, (amin+amax)//2, amin]:
        servo.write(a)
        time.sleep(pause)

# ================== Main Loop ==================
try:
    last_write = 0.0
    WRITE_HZ = 20                 # gentle on Firmata & servos
    writer_index = 0              # for round-robin write order
    last_angles = [0, 0, 0, 0, 0] # for on-screen display

    labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(rgb)
        openness = None

        if result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            landmarks = result.multi_hand_landmarks[0].landmark
            metrics = normalize_metrics(landmarks)

            if open_ref and closed_ref:
                openness = smooth(to_percentages(metrics, open_ref, closed_ref))

                # Rate-limited, staggered writes in rotating order (+2ms gaps)
                now = time.time()
                if now - last_write >= 1.0 / WRITE_HZ:
                    order = [(writer_index + k) % 5 for k in range(5)]
                    for j in order:
                        val = openness[j]
                        amin, amax = PER_FINGER_LIMITS[j]
                        angle = openness_to_angle(val, amin, amax, INVERT[j])
                        servos[j].write(angle)
                        last_angles[j] = int(angle)
                        time.sleep(0.002)  # tiny spacing to avoid buffer hiccups
                    writer_index = (writer_index + 1) % 5
                    last_write = now

        # ---------------- UI Overlay ----------------
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (w - 10, 180), (0, 0, 0), -1)
        y = 40
        cv2.putText(frame, "Keys: O=open calib, C=closed calib, 1..5 sweep servos, Q=quit",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 25
        status = "Calibrated" if open_ref and closed_ref else "Not calibrated"
        color = (0, 255, 0) if open_ref and closed_ref else (0, 255, 255)
        cv2.putText(frame, f"Status: {status}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25

        # Show mapping for sanity
        cv2.putText(frame, "Map: " + " ".join(f"{L}->D{p}" for L, p in zip(labels, SERVO_PINS)),
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1)
        y += 25

        if openness:
            percents = [int(v * 100) for v in openness]
            cv2.putText(frame, "Open%: " + " ".join(f"{L}:{p:3d}%" for L, p in zip(labels, percents)),
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 25
            cv2.putText(frame, "Angles: " + " ".join(f"{L}:{a:3d}°" for L, a in zip(labels, last_angles)),
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

        cv2.imshow("Finger Tracking → Servo Control (Robust)", frame)

        # ---------------- Key Control ----------------
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o') and result.multi_hand_landmarks:
            open_ref = normalize_metrics(result.multi_hand_landmarks[0].landmark)
            for d in smoothers: d.clear()
        elif key == ord('c') and result.multi_hand_landmarks:
            closed_ref = normalize_metrics(result.multi_hand_landmarks[0].landmark)
            for d in smoothers: d.clear()
        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            idx = key - ord('1')  # 0..4 (Thumb..Pinky)
            amin, amax = PER_FINGER_LIMITS[idx]
            sweep(servos[idx], amin, amax)

finally:
    board.exit()
    cap.release()
    cv2.destroyAllWindows()
