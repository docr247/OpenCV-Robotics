import cv2
import mediapipe as mp
import pyfirmata2
import math
import time
from collections import deque

# ================== Hardware Setup ==================
PORT = "/dev/cu.usbmodem1301"   # Change if needed
board = pyfirmata2.Arduino(PORT)

# Thumb, Index, Middle, Ring, Pinky (make sure these match your wiring)
SERVO_PINS = [3, 5, 6, 9, 10]
servos = [board.get_pin(f"d:{p}:s") for p in SERVO_PINS]

# Flip direction per finger if needed (Thumb inverted here)
INVERT = [True, False, False, False, False]

# Per-finger angle limits (tweak as needed)
PER_FINGER_LIMITS = [
    (10, 160),   # Thumb
    (20, 170),   # Index
    (20, 170),   # Middle
    (20, 170),   # Ring
    (20, 170),   # Pinky
]

# ================== Video + MediaPipe ==================
cap = cv2.VideoCapture(0)
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hand = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Landmark indices
TIPS =  [4, 8, 12, 16, 20]
MCPS =  [2, 5, 9, 13, 17]
WRIST = 0
MID_MCP = 9

# Calibration & smoothing
open_ref = None
closed_ref = None
SMOOTH_N = 5
smoothers = [deque(maxlen=SMOOTH_N) for _ in range(5)]

def dist2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def normalize_metrics(landmarks):
    wrist = landmarks[WRIST]
    mid_mcp = landmarks[MID_MCP]
    hand_size = dist2d(wrist, mid_mcp) or 1e-6
    return [dist2d(landmarks[t], landmarks[m]) / hand_size for t, m in zip(TIPS, MCPS)]

def to_percentages(metrics, open_ref, closed_ref):
    out = []
    for i, (m, o, c) in enumerate(zip(metrics, open_ref, closed_ref)):
        # Guard against degenerate calibration (e.g., pinky didn’t change between O and C)
        if abs(o - c) < 1e-3:
            # create a small synthetic span around c so movement still maps
            span = 0.05
            o = c + span
        out.append(clamp01((m - c) / (o - c)))
    return out

def smooth(values):
    smoothed = []
    for i, v in enumerate(values):
        smoothers[i].append(v)
        smoothed.append(sum(smoothers[i]) / len(smoothers[i]))
    return smoothed

def openness_to_angle(value, amin, amax, invert=False):
    value = 1.0 - value if invert else value
    value = clamp01(value)
    return amin + value * (amax - amin)

def sweep(servo, amin, amax, pause=0.3):
    for a in [amin, (amin+amax)//2, amax, (amin+amax)//2, amin]:
        servo.write(a)
        time.sleep(pause)

try:
    last_write = 0
    WRITE_HZ = 30
    last_angles = [0, 0, 0, 0, 0]  # for on-screen display

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

                now = time.time()
                if now - last_write >= 1.0 / WRITE_HZ:
                    for i, val in enumerate(openness):
                        amin, amax = PER_FINGER_LIMITS[i]
                        angle = openness_to_angle(val, amin, amax, INVERT[i])
                        servos[i].write(angle)
                        last_angles[i] = int(angle)
                    last_write = now

        # -------- UI Overlay --------
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (10, 10), (w - 10, 150), (0, 0, 0), -1)
        y = 40
        cv2.putText(frame, "Keys: O=open calib, C=closed calib, 1..5 test servos, Q=quit",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 25
        status = "Calibrated" if open_ref and closed_ref else "Not calibrated"
        color = (0, 255, 0) if open_ref and closed_ref else (0, 255, 255)
        cv2.putText(frame, f"Status: {status}", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y += 25
        if openness:
            labels = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
            percents = [int(v * 100) for v in openness]
            cv2.putText(frame, "Open%: " + " ".join(f"{L}:{p:3d}%" for L, p in zip(labels, percents)),
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            y += 25
            cv2.putText(frame, "Angles: " + " ".join(f"{L}:{a:3d}°" for L, a in zip(labels, last_angles)),
                        (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1)

        cv2.imshow("Finger Tracking → Servo Control", frame)

        # -------- Key Control --------
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
            idx = key - ord('1')  # 0..4
            amin, amax = PER_FINGER_LIMITS[idx]
            sweep(servos[idx], amin, amax)

finally:
    board.exit()
    cap.release()
    cv2.destroyAllWindows()
