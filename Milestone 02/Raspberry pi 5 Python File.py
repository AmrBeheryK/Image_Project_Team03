import cv2
import numpy as np
import subprocess
import serial
import time
from flask import Flask, Response

app = Flask(__name__)

# ── Serial settings ───────────────────────────────────────────────────────────
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 9600
MOTOR_SPEED = 160

# Distance thresholds in cm
NEAR_CM = 20.0
FAR_CM  = 30.0

# Stability
STABLE_FRAMES   = 4
MAX_LOST_FRAMES = 6

# ── Camera settings ───────────────────────────────────────────────────────────
FRAME_WIDTH  = 640
FRAME_HEIGHT = 480
FPS          = 10

# ── Distance calibration (from friend's area-based formula) ──────────────────
# Tune AREA_CALIB_CONST by holding cube at a known distance and adjusting
# until the reading matches. Friend used 60.0 — start here.
AREA_CALIB_CONST = 60.0
KNOWN_WIDTH_CM   = 5.0

# ── Background model ──────────────────────────────────────────────────────────
background = None
BG_FRAMES  = 40
bg_buffer  = []
bg_ready   = False

# ── Brown thresholds for wooden cube ─────────────────────────────────────────
LOWER_BROWN = np.array([5,  50,  30], dtype=np.uint8)
UPPER_BROWN = np.array([30, 255, 255], dtype=np.uint8)

# ── Serial init ───────────────────────────────────────────────────────────────
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2)
    print(f"[OK] Serial connected to {SERIAL_PORT}")
except Exception as e:
    ser = None
    print(f"[WARN] Serial not connected: {e}")

last_cmd = ('S', 0)

def send_cmd(direction, speed):
    global last_cmd
    cmd = (direction, speed)
    if cmd == last_cmd:
        return
    if ser and ser.is_open:
        ser.write(f"{direction},{speed}\n".encode())
    last_cmd = cmd

# ── Milestone transforms ──────────────────────────────────────────────────────

def geo_rotation(gray, angle=15):
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h))

def geo_scaling(gray, factor=0.85):
    h, w = gray.shape[:2]
    scaled = cv2.resize(gray, None, fx=factor, fy=factor, interpolation=cv2.INTER_LINEAR)
    if factor == 1.0:
        return scaled
    if factor < 1.0:
        out = np.zeros_like(gray)
        sh, sw = scaled.shape[:2]
        y0 = (h - sh) // 2
        x0 = (w - sw) // 2
        out[y0:y0+sh, x0:x0+sw] = scaled
        return out
    else:
        sh, sw = scaled.shape[:2]
        y0 = max((sh - h) // 2, 0)
        x0 = max((sw - w) // 2, 0)
        return scaled[y0:y0+h, x0:x0+w]

def intensity_brightness(gray, alpha=1.15, beta=20):
    return cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

def intensity_contrast(gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

# ── Background helpers ────────────────────────────────────────────────────────

def update_background(gray_blur):
    global background, bg_ready, bg_buffer
    if bg_ready:
        return
    bg_buffer.append(gray_blur.astype(np.float32))
    if len(bg_buffer) >= BG_FRAMES:
        background = np.mean(bg_buffer, axis=0).astype(np.uint8)
        bg_ready   = True
        print("[OK] Background model ready.")

def foreground_mask(gray_blur):
    if not bg_ready:
        return None
    diff = cv2.absdiff(gray_blur, background)
    _, mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
    bubble_kernel = np.ones((17, 17), np.uint8)
    mask = cv2.dilate(mask, bubble_kernel, iterations=2)
    mask = cv2.erode(mask,  bubble_kernel, iterations=1)
    close_kernel = np.ones((11, 11), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    return mask

# ── Detection helpers ─────────────────────────────────────────────────────────

def touches_border(x, y, w, h, W, H, margin=5):
    return x <= margin or y <= margin or (x+w) >= (W-margin) or (y+h) >= (H-margin)

def score_contour(cnt, frame_shape):
    H, W = frame_shape[:2]
    hull = cv2.convexHull(cnt)
    area = cv2.contourArea(hull)

    if area < 800 or area > 150000:
        return None

    x, y, w, h = cv2.boundingRect(hull)
    if w <= 0 or h <= 0:
        return None
    if touches_border(x, y, w, h, W, H, margin=5):
        return None

    rect = cv2.minAreaRect(hull)
    (cX, cY), (rw, rh), _ = rect
    if rw <= 0 or rh <= 0:
        return None

    ratio     = max(rw, rh) / max(1.0, min(rw, rh))
    rect_area = rw * rh
    solidity  = area / max(rect_area, 1.0)
    extent    = area / max(float(w * h), 1.0)

    if ratio > 2.2:   return None
    if solidity < 0.45: return None
    if extent < 0.25:   return None

    frame_center = np.array([W / 2.0, H / 2.0])
    obj_center   = np.array([cX, cY])
    center_dist  = np.linalg.norm(obj_center - frame_center)

    score = (
        2.0   * area +
        1500.0 * solidity +
        1000.0 * extent -
        6.0   * center_dist -
        500.0 * abs(ratio - 1.0)
    )

    return {"score": score, "hull": hull, "rect": rect,
            "center": (int(cX), int(cY)), "area": area}

def choose_best_contour(contours, frame_shape):
    best = None
    best_score = -1e18
    for cnt in contours:
        result = score_contour(cnt, frame_shape)
        if result is None:
            continue
        if result["score"] > best_score:
            best_score = result["score"]
            best = result
    return best

def detect_cube(frame):
    gray        = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_for_bg = intensity_contrast(gray)
    blur        = cv2.GaussianBlur(gray_for_bg, (13, 13), 0)

    update_background(blur)

    if not bg_ready:
        return {"status": "calibrating",
                "cam_view": frame.copy(),
                "mask_view": cv2.cvtColor(blur, cv2.COLOR_GRAY2BGR),
                "found": False}

    fg = foreground_mask(blur)
    if fg is None:
        return {"status": "not_found",
                "cam_view": frame.copy(),
                "mask_view": cv2.cvtColor(np.zeros_like(gray), cv2.COLOR_GRAY2BGR),
                "found": False}

    hsv        = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brown_mask = cv2.inRange(hsv, LOWER_BROWN, UPPER_BROWN)
    cube_mask  = cv2.bitwise_and(brown_mask, fg)

    kernel5 = np.ones((5, 5), np.uint8)
    kernel9 = np.ones((9, 9), np.uint8)
    cube_mask = cv2.morphologyEx(cube_mask, cv2.MORPH_CLOSE, kernel9, iterations=2)
    cube_mask = cv2.morphologyEx(cube_mask, cv2.MORPH_OPEN,  kernel5, iterations=1)
    cube_mask = cv2.dilate(cube_mask, kernel5, iterations=1)

    cam_view  = frame.copy()
    mask_view = cv2.cvtColor(cube_mask, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(cube_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best   = choose_best_contour(contours, frame.shape)
    method = "FG + BROWN"

    if best is None:
        fg_only = fg.copy()
        fg_only = cv2.morphologyEx(fg_only, cv2.MORPH_CLOSE, kernel9, iterations=2)
        fg_only = cv2.morphologyEx(fg_only, cv2.MORPH_OPEN,  kernel5, iterations=1)
        contours, _ = cv2.findContours(fg_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best      = choose_best_contour(contours, frame.shape)
        mask_view = cv2.cvtColor(fg_only, cv2.COLOR_GRAY2BGR)
        method    = "FG ONLY"

    if best is None:
        return {"status": "not_found", "cam_view": cam_view,
                "mask_view": mask_view, "found": False}

    hull   = best["hull"]
    rect   = best["rect"]
    center = best["center"]
    area   = best["area"]

    box = np.int32(cv2.boxPoints(rect))

    # ── Friend's area-based distance formula ─────────────────────────
    dist_cm = AREA_CALIB_CONST * (KNOWN_WIDTH_CM / np.sqrt(area / 100))

    for view in [cam_view, mask_view]:
        cv2.drawContours(view, [hull], 0, (255, 0, 0), 2)
        cv2.drawContours(view, [box],  0, (0, 255, 0), 3)
        cv2.circle(view, center, 5, (0, 0, 255), -1)
        top_corner = tuple(box[box[:, 1].argmin()])
        cv2.putText(view, f"CUBE: {dist_cm:.1f} cm",
                    (top_corner[0], max(20, top_corner[1]-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(view, f"Method: {method}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    return {"status": "found", "cam_view": cam_view,
            "mask_view": mask_view, "found": True, "distance": dist_cm}

# ── Frame generator ───────────────────────────────────────────────────────────

def generate_frames():
    pipe = subprocess.Popen(
        ['rpicam-vid', '-t', '0', '--inline',
         '--width',  str(FRAME_WIDTH),
         '--height', str(FRAME_HEIGHT),
         '--framerate', str(FPS),
         '--codec', 'mjpeg', '-o', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
    )

    data          = b''
    pending_cmd   = ('S', 0)
    pending_count = 0
    smoothed_dist = None
    lost_frames   = 0

    while True:
        # Read large chunk, always use latest frame
        data += pipe.stdout.read(65536)

        last_a = data.rfind(b'\xff\xd8')
        last_b = data.rfind(b'\xff\xd9')

        if last_a == -1 or last_b == -1 or last_a >= last_b:
            continue

        jpg  = data[last_a: last_b + 2]
        data = data[last_b + 2:]

        frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            continue
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Milestone transforms (computed, available for report)
        gray_tmp  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _rotated  = geo_rotation(gray_tmp,        angle=15)
        _scaled   = geo_scaling(gray_tmp,         factor=0.85)
        _bright   = intensity_brightness(gray_tmp, alpha=1.15, beta=20)
        _contrast = intensity_contrast(gray_tmp)
        _smooth   = cv2.GaussianBlur(_contrast, (5, 5), 0)

        result   = detect_cube(frame)
        cam_view  = result["cam_view"]
        mask_view = result["mask_view"]

        desired_cmd = ('S', 0)

        if result["status"] == "calibrating":
            remaining = BG_FRAMES - len(bg_buffer)
            for view in [cam_view, mask_view]:
                cv2.putText(view, f"Calibrating... ({remaining} frames)",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                cv2.putText(view, "KEEP CUBE OUT OF FRAME",
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            desired_cmd = ('S', 0)

        elif result["status"] == "not_found":
            lost_frames += 1
            if lost_frames >= MAX_LOST_FRAMES:
                smoothed_dist = None
            for view in [cam_view, mask_view]:
                cv2.putText(view, "Cube not found",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            desired_cmd = ('S', 0)

        elif result["status"] == "found":
            lost_frames = 0
            dist_cm     = result["distance"]

            # Smooth the distance reading
            if smoothed_dist is None:
                smoothed_dist = dist_cm
            else:
                smoothed_dist = 0.75 * smoothed_dist + 0.25 * dist_cm

            if smoothed_dist <= NEAR_CM:
                desired_cmd  = ('F', MOTOR_SPEED)
                motor_text   = f"MOTOR: CW  | {smoothed_dist:.1f} cm"
                motor_color  = (0, 255, 255)
            elif smoothed_dist >= FAR_CM:
                desired_cmd  = ('B', MOTOR_SPEED)
                motor_text   = f"MOTOR: CCW | {smoothed_dist:.1f} cm"
                motor_color  = (0, 165, 255)
            else:
                desired_cmd  = ('S', 0)
                motor_text   = f"MOTOR: STOP | {smoothed_dist:.1f} cm"
                motor_color  = (200, 200, 200)

            for view in [cam_view, mask_view]:
                cv2.putText(view, motor_text,
                            (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, motor_color, 2)

        # Stability gate
        if desired_cmd == pending_cmd:
            pending_count += 1
        else:
            pending_cmd   = desired_cmd
            pending_count = 1

        if pending_count >= STABLE_FRAMES:
            send_cmd(pending_cmd[0], pending_cmd[1])

        # Crosshair
        h, w = cam_view.shape[:2]
        cv2.line(cam_view, (w//2-10, h//2), (w//2+10, h//2), (255,255,255), 1)
        cv2.line(cam_view, (w//2, h//2-10), (w//2, h//2+10), (255,255,255), 1)

        combined = np.hstack((cam_view, mask_view))
        cv2.putText(combined, "LIVE CAMERA",    (10,              30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(combined, "DETECTION MASK", (FRAME_WIDTH + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        ok, buffer = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            continue

        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ── Flask ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return "<h1>Cube Detection + Motor Control</h1><img src='/video_feed'>"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
