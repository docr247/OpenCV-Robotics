"""
Finger Tracking → 5-Servo Robotic Hand Control
Author: Ravi Suppiah (refactored by ChatGPT)
Dependencies: opencv-python, mediapipe, pyserial
"""

import cv2
import mediapipe as mp
import serial
import math
import time
import glob
from collections import deque

# ================== Hardware Setup ==================
PORT = "/dev/cu.usbmodem1401"   # Preferred port. Auto-detect fallback is used if this path is missing.

def resolve_port(preferred_port):
    if preferred_port and glob.glob(preferred_port):
        return preferred_port
    candidates = sorted(glob.glob("/dev/cu.usbmodem*"))
    return candidates[0] if candidates else None

BAUD_RATE = 115200
RECONNECT_INTERVAL = 1.0
arduino = None
last_connect_attempt = 0.0

def connect_arduino():
    global arduino, last_connect_attempt
    now = time.time()
    if now - last_connect_attempt < RECONNECT_INTERVAL:
        return False
    last_connect_attempt = now

    port = resolve_port(PORT)
    if not port:
        return False

    try:
        arduino = serial.Serial(port, BAUD_RATE, timeout=1, write_timeout=1)
        time.sleep(2.0)  # The Uno resets when the serial connection opens.
        arduino.reset_input_buffer()
        print(f"Connected to Arduino on {port}")
        return True
    except serial.SerialException as exc:
        arduino = None
        print(f"Arduino connection failed: {exc}")
        return False

connect_arduino()

# Thumb, Index, Middle, Ring, Pinky  (must match TIPS/MCPS order)
SERVO_PINS = [10, 9, 6, 5, 3]

# Flip only the thumb if needed
INVERT = [True, False, False, False, False]

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
THUMB_RESPONSE_GAIN = 1.5

# ================== Helper Functions ==================
def dist2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def normalize_metrics(landmarks):
    """Return per-finger normalized metric (tip↔MCP distance / hand_size)."""
    wrist = landmarks[WRIST]
    mid_mcp = landmarks[MID_MCP]
    hand_size = dist2d(wrist, mid_mcp) or 1e-6
    thumb_openness = dist2d(landmarks[4], landmarks[5]) / hand_size
    finger_openness = [
        dist2d(landmarks[t], landmarks[m]) / hand_size
        for t, m in zip(TIPS[1:], MCPS[1:])
    ]
    return [thumb_openness, *finger_openness]

def to_percentages(metrics, open_ref, closed_ref):
    """Map each metric to openness 0..1 using calibration data."""
    out = []
    for m, o, c in zip(metrics, open_ref, closed_ref):
        out.append(clamp01((m - c) / (o - c))) if abs(o - c) > 1e-6 else out.append(0.0)
    out[0] = clamp01(0.5 + (out[0] - 0.5) * THUMB_RESPONSE_GAIN)
    return out

def smooth(values):
    """Moving-average smoothing per finger."""
    smoothed = []
    for i, v in enumerate(values):
        smoothers[i].append(v)
        smoothed.append(sum(smoothers[i]) / len(smoothers[i]))
    return smoothed

def openness_to_angle(value, amin, amax, invert=False):
    """Map openness [0,1] → servo angle [amin,amax]."""
    value = 1.0 - value if invert else value
    value = clamp01(value)
    return amin + value * (amax - amin)

def send_angles(angles):
    """Send one complete thumb-to-pinky angle frame to the Uno."""
    global arduino
    if arduino is None and not connect_arduino():
        return

    payload = "A," + ",".join(str(int(round(angle))) for angle in angles) + "\n"
    try:
        arduino.write(payload.encode("ascii"))
    except serial.SerialException as exc:
        print(f"Arduino disconnected: {exc}")
        arduino.close()
        arduino = None

# ================== Main Loop ==================
try:
    last_write = 0
    WRITE_HZ = 30  # Max servo update rate

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hand.process(rgb)
        openness = None

        if result.multi_hand_landmarks:
            landmarks = result.multi_hand_landmarks[0].landmark
            metrics = normalize_metrics(landmarks)
            mp_drawing.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)

            if open_ref and closed_ref:
                openness = smooth(to_percentages(metrics, open_ref, closed_ref))

                # Rate-limited servo updates
                now = time.time()
                if now - last_write >= 1.0 / WRITE_HZ:
                    next_angles = []
                    for i, val in enumerate(openness):
                        amin, amax = PER_FINGER_LIMITS[i]
                        angle = openness_to_angle(val, amin, amax, INVERT[i])
                        next_angles.append(angle)
                    send_angles(next_angles)
                    last_write = now

        # ============ On-screen UI ============
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (w - 10, 110), (0, 0, 0), -1)
        y = 40
        cv2.putText(frame, "Keys: O=open calib, C=closed calib, Q=quit", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y += 25
        status = "Calibrated" if open_ref and closed_ref else "Not calibrated"
        color = (0, 255, 0) if open_ref and closed_ref else (0, 255, 255)
        cv2.putText(frame, f"Status: {status}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25
        if openness:
            labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            percents = [int(v * 100) for v in openness]
            cv2.putText(frame, "Open%: " + " ".join(f"{L}:{p:3d}" for L, p in zip(labels, percents)),
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Finger Tracking → Servo Control", frame)

        # ============ Key Control ============
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('o') and result.multi_hand_landmarks:
            open_ref = normalize_metrics(result.multi_hand_landmarks[0].landmark)
            for d in smoothers: d.clear()
        elif key == ord('c') and result.multi_hand_landmarks:
            closed_ref = normalize_metrics(result.multi_hand_landmarks[0].landmark)
            for d in smoothers: d.clear()

finally:
    if arduino is not None:
        arduino.close()
    hand.close()
    cap.release()
    cv2.destroyAllWindows()
