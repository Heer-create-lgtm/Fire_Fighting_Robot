#!/usr/bin/env python3
import os
import serial
import time
import re
import threading
import traceback
import collections
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
import cv2
import numpy as np
import requests  # still used for MJPEG fallback
from ultralytics import YOLO
import playsound
import socket

# ----------------- CONFIG (env override) -----------------
RECEIVER_HOST = os.environ.get("RECEIVER_HOST", "192.168.137.51")
RECEIVER_PORT = int(os.environ.get("RECEIVER_PORT", "5000"))
HTTP_TIMEOUT = float(os.environ.get("RECEIVER_TIMEOUT", "1.5"))   # socket timeout
HTTP_RETRIES = int(os.environ.get("HTTP_RETRIES", "2"))          # number of retries (total attempts = retries+1)
HTTP_RETRY_HOLD = float(os.environ.get("HTTP_RETRY_HOLD", "1.0"))# wait on failure before retry
RETRY_WAIT = float(os.environ.get("HTTP_RETRY_HOLD", "1.0"))    # add this near the top with other config vars
STEP_WAIT = float(os.environ.get("HTTP_STEP_WAIT", "1.0"))       # wait after success before next command

# ML / audio / serial config (defaults preserved)
THRESHOLD = float(os.environ.get("THRESHOLD", "0.9"))
ALERT_SOUND_PATH = os.environ.get("ALERT_SOUND_PATH", r"C:/Users/shrey/Downloads/alert.mp3")
MODEL_PATH = os.environ.get("YOLO_MODEL", r"C:/Users/shrey/OneDrive/Desktop/best.pt")

ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "COM3")
ARDUINO_BAUD = int(os.environ.get("ARDUINO_BAUD", "9600"))

ESP32_URL = os.environ.get("ESP32_STREAM_URL", "http://192.168.137.245:4747/video")
ANNOTATED_DIR = os.path.abspath("annotated")
os.makedirs(ANNOTATED_DIR, exist_ok=True)

LOWRES_IMG_SIZE = int(os.environ.get("LOWRES_IMG_SIZE", "320"))
HIGHRES_IMG_SIZE = int(os.environ.get("HIGHRES_IMG_SIZE", "640"))
HIGHRES_INTERVAL = float(os.environ.get("HIGHRES_INTERVAL", "2.0"))
INFERENCE_INTERVAL = float(os.environ.get("INFERENCE_INTERVAL", "0.25"))

YOLO_CONF_LOWRES = float(os.environ.get("YOLO_CONF_LOWRES", "0.06"))
YOLO_CONF_HIGHRES = float(os.environ.get("YOLO_CONF_HIGHRES", "0.12"))
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.45"))

HSV_MIN_MEAN_V = float(os.environ.get("HSV_MIN_MEAN_V", "40"))
FIRE_COLOR_MIN_FRACTION = float(os.environ.get("FIRE_COLOR_MIN_FRACTION", "0.04"))
FLICKER_STDDEV_MIN = float(os.environ.get("FLICKER_STDDEV_MIN", "6.0"))

HSV_WEIGHT = float(os.environ.get("HSV_WEIGHT", "0.14"))
FLICK_WEIGHT = float(os.environ.get("FLICK_WEIGHT", "0.08"))
SCORE_REQUIRED = float(os.environ.get("SCORE_REQUIRED", "0.28"))

HISTORY_LEN = int(os.environ.get("HISTORY_LEN", "6"))
DEBUG = os.environ.get("DEBUG", "1") not in ("0", "false", "False")
ENABLE_HTTP_SERVER = os.environ.get("ENABLE_HTTP", "1") not in ("0", "false", "False")
HTTP_PORT = int(os.environ.get("HTTP_PORT", "8080"))
ANNOT_SAVE_PERIOD = float(os.environ.get("ANNOT_SAVE_PERIOD", "1.0"))

CAM_VERTICAL_FOV_DEG = 40.0
DISTANCE_REAL_HEIGHT_M = 1.0

# ----------------- GLOBALS -----------------
_frame_history = collections.deque(maxlen=HISTORY_LEN)
_latest_lock = threading.Lock()
_stop_event = threading.Event()

ml_disabled = False
last_distance_m = None
last_angle = -1

# Setup Arduino serial (wrapped in try)
arduino = None
try:
    arduino = serial.serialwin32.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=0.1)
    time.sleep(2)
    print(f"[SERIAL] opened {ARDUINO_PORT} @ {ARDUINO_BAUD}")
except Exception as e:
    print("[SERIAL] could not open serial port:", e)
    arduino = None

# ----------------- helper: raw socket GET -----------------
def raw_get(path: str):
    host = RECEIVER_HOST
    port = RECEIVER_PORT
    path = "/" + path.lstrip("/")

    # Long timeout for /fire 
    is_fire = path.startswith("/fire")
    timeout = 1.5 if not is_fire else 120.0   # 🔥 Increase timeout only for fire
    retries = HTTP_RETRIES if not is_fire else 1   # 🔥 never retry fire

    req = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}\r\n"
        f"Connection: close\r\n\r\n"
    ).encode()

    for attempt in range(1, retries + 1):
        print(f"[RAW-HTTP] attempt {attempt}: {host}:{port}{path}")

        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(timeout)
            s.connect((host, port))
            s.sendall(req)

            data = s.recv(1024)
            s.close()

            if not data:
                if is_fire:
                    print("[RAW-HTTP] /fire got empty but connection OK → treat as success.")
                    return True
                print("[RAW-HTTP] no response (empty).")
                time.sleep(RETRY_WAIT)
                continue

            header = data.split(b"\r\n")[0].decode(errors="ignore")
            print("[RAW-HTTP] RESPONSE:", header)

            if "200" in header:
                return True

        except Exception as e:
            print("[RAW-HTTP] ERROR:", str(e))
            if is_fire:
                print("[RAW-HTTP] /fire error, but we DO NOT retry. Treat as success.")
                return True

        print(f"[RAW-HTTP] holding {RETRY_WAIT}s before retry...")

    return False

# ----------------- FIRE SEQUENCE -----------------
def fire_sequence(angle, forward_distance=50):
    print("\n================ FIRE SEQUENCE ================\n")

    try:
        angle = int(angle)
    except:
        angle = -1

    # -------- DETERMINE TURN DIRECTION --------
    if 0 <= angle <= 180:
        turn_path = f"left?angle={angle}"
        print(f"[FIRE] Using LEFT turn → angle {angle}")
    else:
        # Convert big angle into right turn
        right_angle = 360 - angle
        turn_path = f"right?angle={right_angle}"
        print(f"[FIRE] Using RIGHT turn → angle {right_angle}")

    # -------- SEND TURN COMMAND --------
    ok = raw_get(turn_path)
    status = "ok" if ok else "failed"
    print(f"[FIRE] TURN -> {ok} ({status})")

    if not ok:
        print("[FIRE] TURN failed — aborting sequence.")
        return False

    print(f"[FIRE] TURN success → waiting 1.0s\n")
    time.sleep(1.0)

    # -------- FORWARD 50 --------
    fwd_path = f"forward?distance={forward_distance}"
    print(f"[FIRE] Sending FORWARD → {fwd_path}")

    ok = raw_get(fwd_path)
    status = "ok" if ok else "failed"
    print(f"[FIRE] FORWARD -> {ok} ({status})")

    if not ok:
        print("[FIRE] FORWARD failed — aborting sequence.")
        return False

    print(f"[FIRE] FORWARD success → waiting 1.0s\n")
    time.sleep(1.0)

    # -------- FIRE --------
    print("[FIRE] Sending FIRE → fire")

    ok = raw_get("fire")
    status = "ok" if ok else "failed"
    print(f"[FIRE] FIRE -> {ok} ({status})")

    if not ok:
        print("[FIRE] FIRE failed — aborting sequence.")
        return False

    print("\n🔥 FIRE SEQUENCE COMPLETE — SHUTTING DOWN ML 🔥\n")
    _stop_event.set()
    return True

# ----------------- audio -----------------
def play_alert_sound():
    try:
        if ALERT_SOUND_PATH:
            threading.Thread(target=playsound.playsound, args=(ALERT_SOUND_PATH,), kwargs={'block': False}, daemon=True).start()
    except Exception as e:
        if DEBUG: print("[AUDIO] error:", e)

# ----------------- utilities -----------------
def log(*a, **k):
    if DEBUG: print(*a, **k)

def hsv_fire_fraction(bgr_roi):
    if bgr_roi is None or bgr_roi.size == 0:
        return 0.0, 0.0
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    mean_v = float(np.mean(v))
    lower1 = np.array([0, 90, 60]); upper1 = np.array([25, 255, 255])
    lower2 = np.array([170, 90, 60]); upper2 = np.array([179, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower1, upper1), cv2.inRange(hsv, lower2, upper2))
    frac = float(np.count_nonzero(mask)) / (mask.size + 1e-9)
    return frac, mean_v

def compute_flicker_for_box(box):
    if len(_frame_history) < 2:
        return 0.0
    x1,y1,x2,y2 = map(int, box)
    vals = []
    for hsv in _frame_history:
        rows, cols = hsv.shape[:2]
        cx1 = max(0, min(cols-1, x1)); cx2 = max(0, min(cols, x2))
        cy1 = max(0, min(rows-1, y1)); cy2 = max(0, min(rows, y2))
        if cy2 <= cy1 or cx2 <= cx1: continue
        roi_v = hsv[cy1:cy2, cx1:cx2, 2]
        if roi_v.size == 0: continue
        vals.append(float(np.mean(roi_v)))
    if not vals: return 0.0
    return float(np.std(vals))

def save_annotated(img, prefix="latest"):
    try:
        tmp = os.path.join(ANNOTATED_DIR, f".{prefix}.tmp.jpg")
        out = os.path.join(ANNOTATED_DIR, f"{prefix}.jpg")
        cv2.imwrite(tmp, img)
        os.replace(tmp, out)
    except Exception as e:
        log("[SAVE]", e)
    now = time.time()
    if not hasattr(save_annotated, "_last"):
        save_annotated._last = 0.0
    if now - save_annotated._last >= ANNOT_SAVE_PERIOD:
        stamped = os.path.join(ANNOTATED_DIR, f"frame_{int(now)}.jpg")
        try:
            cv2.imwrite(stamped, img)
            save_annotated._last = now
            log("[SAVE] wrote", stamped)
        except Exception as e:
            log("[SAVE] stamped:", e)

# ----------------- stream frames -----------------
def stream_frames_to_queue(frame_queue):
    url = ESP32_URL
    log("[STREAM] VideoCapture on", url)
    try:
        cap = cv2.VideoCapture(url)
    except Exception as e:
        log("[STREAM] VideoCapture exception:", e)
        cap = None

    if cap is not None and cap.isOpened():
        try:
            while not _stop_event.is_set():
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue
                with _latest_lock:
                    frame_queue.clear()
                    frame_queue.append(frame)
        except Exception as e:
            log("[STREAM] read error:", e)
            traceback.print_exc()
        finally:
            try: cap.release()
            except: pass
        log("[STREAM] ended")
        return

    # fallback: simple requests MJPEG
    log("[STREAM] falling back to requests MJPEG for", url)
    try:
        r = requests.get(url, stream=True, timeout=(5,10))
    except Exception as e:
        log("[STREAM] connection failed:", e)
        return
    buf = bytearray()
    try:
        for chunk in r.iter_content(chunk_size=4096):
            if _stop_event.is_set(): break
            if not chunk: continue
            buf.extend(chunk)
            a = bytes(buf).find(b'\xff\xd8')
            b = bytes(buf).find(b'\xff\xd9', a+2) if a != -1 else -1
            if a != -1 and b != -1:
                jpg = bytes(buf[a:b+2]); buf = buf[b+2:]
                arr = np.frombuffer(jpg, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                with _latest_lock:
                    frame_queue.clear()
                    frame_queue.append(frame)
    except Exception as e:
        log("[STREAM] mjpeg read error:", e)
    finally:
        try: r.close()
        except: pass
    log("[STREAM] mjpeg ended")

# ----------------- distance estimator -----------------
def estimate_distance_from_bbox(frame_h, bbox_h_pixels, cam_vfov_deg=CAM_VERTICAL_FOV_DEG, object_height_m=DISTANCE_REAL_HEIGHT_M):
    if bbox_h_pixels <= 0: return None
    v_fov_rad = np.deg2rad(cam_vfov_deg)
    f_pixels = (frame_h / 2.0) / np.tan(v_fov_rad / 2.0)
    dist_m = (object_height_m * f_pixels) / float(bbox_h_pixels)
    return float(dist_m)

# ----------------- main inference loop -----------------
def inference_loop(model):
    global ml_disabled, last_distance_m, last_angle

    frame_queue = collections.deque(maxlen=1)
    threading.Thread(target=stream_frames_to_queue, args=(frame_queue,), daemon=True).start()

    last_highres = 0.0
    last_infer = 0.0
    frame_count = 0

    while not _stop_event.is_set():
        try:
            now = time.time()
            if now - last_infer < INFERENCE_INTERVAL:
                time.sleep(0.005)
                continue
            last_infer = time.time()

            with _latest_lock:
                if not frame_queue:
                    time.sleep(0.005)
                    continue
                frame = frame_queue[-1].copy()

            # read Arduino serial (angle)
            if arduino and arduino.in_waiting:
                raw = arduino.readline().decode(errors='ignore').strip()
                if raw:
                    if DEBUG: print("[SERIAL RAW]", raw)
                    m = re.search(r'ANGLE[:\s]*(-?\d+)', raw)
                    if m:
                        try:
                            angle_val = int(m.group(1))
                            last_angle = angle_val
                            if DEBUG: print("[SERIAL] parsed ANGLE ->", last_angle)
                        except Exception:
                            if DEBUG: print("[SERIAL] failed to int() the ANGLE value:", m.group(1))

            frame_count += 1
            h, w = frame.shape[:2]

            hsv_full = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            _frame_history.append(hsv_full)

            detections = []

            if ml_disabled:
                if arduino:
                    try: arduino.write(b"STOP\n")
                    except: pass
                time.sleep(0.05)
                continue

            # low-res YOLO on full frame
            try:
                small = cv2.resize(frame, (LOWRES_IMG_SIZE, LOWRES_IMG_SIZE), interpolation=cv2.INTER_AREA)
                if DEBUG: log(f"[INFER] low-res imgsz={LOWRES_IMG_SIZE}")
                t0 = time.time()
                res = model.predict(small, imgsz=LOWRES_IMG_SIZE, conf=YOLO_CONF_LOWRES, iou=YOLO_IOU, verbose=False)
                if DEBUG: log("[INFER] low-res time:", time.time()-t0)
                r0 = res[0] if res and len(res) > 0 else None
                if r0 is not None and hasattr(r0, "boxes") and r0.boxes is not None:
                    for b in r0.boxes.data:
                        x1,y1,x2,y2,conf,cls = map(float, b[:6])
                        sx = w / float(small.shape[1]); sy = h / float(small.shape[0])
                        x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy
                        x1,y1,x2,y2 = max(0,int(x1)), max(0,int(y1)), min(w-1,int(x2)), min(h-1,int(y2))
                        if x2 <= x1 or y2 <= y1: continue
                        roi = frame[y1:y2, x1:x2]
                        hsv_frac, mean_v = hsv_fire_fraction(roi)
                        flick = compute_flicker_for_box([x1,y1,x2,y2])
                        score = conf + (HSV_WEIGHT if hsv_frac >= FIRE_COLOR_MIN_FRACTION else 0.0) + (FLICK_WEIGHT if flick >= FLICKER_STDDEV_MIN else 0.0)
                        detections.append({'box':[x1,y1,x2,y2], 'conf':conf, 'score':score})
            except Exception as e:
                log("[INFER] low-res error:", e)

            # occasional high-res pass
            if time.time() - last_highres >= HIGHRES_INTERVAL:
                try:
                    big = cv2.resize(frame, (HIGHRES_IMG_SIZE, HIGHRES_IMG_SIZE), interpolation=cv2.INTER_AREA)
                    if DEBUG: log(f"[INFER] high-res imgsz={HIGHRES_IMG_SIZE}")
                    t0 = time.time()
                    res = model.predict(big, imgsz=HIGHRES_IMG_SIZE, conf=YOLO_CONF_HIGHRES, iou=YOLO_IOU, verbose=False, augment=True)
                    if DEBUG: log("[INFER] high-res time:", time.time()-t0)
                    r0 = res[0] if res and len(res) > 0 else None
                    if r0 is not None and hasattr(r0, "boxes") and r0.boxes is not None:
                        for b in r0.boxes.data:
                            x1,y1,x2,y2,conf,cls = map(float, b[:6])
                            sx = w / float(big.shape[1]); sy = h / float(big.shape[0])
                            x1 *= sx; x2 *= sx; y1 *= sy; y2 *= sy
                            x1,y1,x2,y2 = max(0,int(x1)), max(0,int(y1)), min(w-1,int(x2)), min(h-1,int(y2))
                            if x2 <= x1 or y2 <= y1: continue
                            roi = frame[y1:y2, x1:x2]
                            hsv_frac, mean_v = hsv_fire_fraction(roi)
                            flick = compute_flicker_for_box([x1,y1,x2,y2])
                            score = conf + (HSV_WEIGHT if hsv_frac >= FIRE_COLOR_MIN_FRACTION else 0.0) + (FLICK_WEIGHT if flick >= FLICKER_STDDEV_MIN else 0.0)
                            detections.append({'box':[x1,y1,x2,y2], 'conf':conf, 'score':score})
                    last_highres = time.time()
                except Exception as e:
                    log("[INFER] high-res error:", e)

            # fallback coarse HSV
            if len(detections) == 0:
                cols = max(3, w // 160)
                rows = max(3, h // 120)
                win_w = max(32, w // cols)
                win_h = max(32, h // rows)
                found_fallback = 0
                for gx in range(0, w, win_w):
                    for gy in range(0, h, win_h):
                        x1 = gx; y1 = gy; x2 = min(w, gx + win_w); y2 = min(h, gy + win_h)
                        roi = frame[y1:y2, x1:x2]
                        if roi.size == 0: continue
                        hsv_frac, mean_v = hsv_fire_fraction(roi)
                        if hsv_frac >= FIRE_COLOR_MIN_FRACTION and mean_v >= HSV_MIN_MEAN_V:
                            flick = compute_flicker_for_box([x1,y1,x2,y2])
                            if flick >= FLICKER_STDDEV_MIN:
                                score = HSV_WEIGHT + FLICK_WEIGHT
                                detections.append({'box':[x1,y1,x2,y2], 'conf':0.0, 'score':score})
                                found_fallback += 1
                                if DEBUG: log(f"[FALLBACK] {x1},{y1},{x2},{y2} hsv={hsv_frac:.3f} flick={flick:.3f}")
                        if found_fallback >= 6:
                            break
                    if found_fallback >= 6:
                        break

            # fusion/dedupe (NMS-like)
            final_detections = [d for d in detections if d['score'] >= SCORE_REQUIRED]
            final_sorted = sorted(final_detections, key=lambda x: -x['score'])
            kept = []
            for cand in final_sorted:
                x1,y1,x2,y2 = cand['box']
                keep = True
                for k in kept:
                    xx1 = max(x1, k['box'][0]); yy1 = max(y1, k['box'][1])
                    xx2 = min(x2, k['box'][2]); yy2 = min(y2, k['box'][3])
                    iw = max(0, xx2 - xx1); ih = max(0, yy2 - yy1)
                    inter = iw * ih
                    area_a = (x2 - x1) * (y2 - y1)
                    area_b = (k['box'][2] - k['box'][0]) * (k['box'][3] - k['box'][1])
                    union = area_a + area_b - inter + 1e-9
                    iou = inter / union
                    if iou > 0.4:
                        keep = False
                        break
                if keep:
                    kept.append(cand)
            confirmed = kept

            # compute highest score & distance
            fire_score = 0
            last_distance_m = None
            if confirmed:
                fire_score = max(d['score'] for d in confirmed)
                best = max(confirmed, key=lambda x: x['score'])
                x1,y1,x2,y2 = best['box']
                bbox_h = float(y2 - y1)
                last_distance_m = estimate_distance_from_bbox(h, bbox_h)

            # FIRE logic using raw socket sequence
            if fire_score >= THRESHOLD and not ml_disabled:
                print(f"\n🔥 FIRE DETECTED | Score={fire_score:.3f}\n")
                if arduino:
                    try: arduino.write(b"STOP\n")
                    except: pass

                # Try to fetch fresh angle if last_angle invalid
                angle_to_send = last_angle
                if angle_to_send is None or int(angle_to_send) < 0:
                    print("[FIRE] Waiting for valid ANGLE from Arduino...")
                    start = time.time()
                    while time.time() - start < 0.6:
                        if arduino and arduino.in_waiting:
                            line = arduino.readline().decode(errors='ignore').strip()
                            if line:
                                if DEBUG: print("[SERIAL RAW]", line)
                                m = re.search(r'(-?\d+)', line)
                                if m:
                                    try:
                                        angle_to_send = int(m.group(1))
                                        last_angle = angle_to_send
                                        print("[FIRE] FINAL ANGLE =", angle_to_send)
                                        break
                                    except:
                                        pass
                        time.sleep(0.01)

                # if still invalid, abort safely
                try:
                    if angle_to_send is None or int(angle_to_send) < 0:
                        print("[FIRE] could not obtain valid angle -> aborting fire sequence to avoid negative durations on robot")
                        ml_disabled = True
                        # do not send commands; just disable ML
                        _stop_event.set()
                        break
                except:
                    print("[FIRE] angle parse error -> abort")
                    ml_disabled = True
                    _stop_event.set()
                    break

                # ensure ml disabled to prevent race
                ml_disabled = True
                print("[FIRE] FINAL ANGLE ->", angle_to_send)
                # run sequence (blocking) — it will set _stop_event on success
                seq_ok = fire_sequence(angle_to_send, forward_distance=50)
                if not seq_ok:
                    print("[LOOP] fire sequence aborted (one of the commands failed). ML disabled.")
                    # leave ml_disabled True; stop loop
                    _stop_event.set()
                    break
                # sequence success => _stop_event already set inside fire_sequence
                break

            # annotate & save
            annotated = frame.copy()
            cv2.putText(annotated, f"score={fire_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            if last_distance_m is not None:
                cv2.putText(annotated, f"Dist(m): {last_distance_m:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            save_annotated(annotated)

            if DEBUG and frame_count % 5 == 0:
                log(f"[STAT] frame={frame_count} detections={len(confirmed)} fire_score={fire_score:.3f} dist={last_distance_m}")

        except Exception as e:
            log("[LOOP] unexpected:", e)
            traceback.print_exc()
            time.sleep(0.02)

    log("[LOOP] exiting (ML disabled or stop event)")

# ----------------- MAIN -----------------
def main():
    try:
        log("Loading YOLO model from:", MODEL_PATH)
        model = YOLO(MODEL_PATH)
        log("[MAIN] model loaded; names:", getattr(model.model, "names", None) or getattr(model, "names", None))
    except Exception as e:
        print("[MAIN] could not load model:", e)
        traceback.print_exc()
        return

    server = None
    if ENABLE_HTTP_SERVER:
        handler = partial(SimpleHTTPRequestHandler, directory=ANNOTATED_DIR)
        server = ThreadingHTTPServer(("0.0.0.0", HTTP_PORT), handler)
        th = threading.Thread(target=server.serve_forever, daemon=True)
        th.start()
        log(f"[HTTP] serving {ANNOTATED_DIR} on {HTTP_PORT}")

    try:
        inference_loop(model)
    except KeyboardInterrupt:
        print("[MAIN] keyboard interrupt")
    finally:
        _stop_event.set()
        if server:
            try: server.shutdown()
            except: pass
        try:
            if arduino: arduino.close()
        except: pass
        print("[MAIN] exiting")

if __name__ == "__main__":
    main()