from flask import Flask, request
import RPi.GPIO as GPIO
import threading
import pigpio
import time
import math
import serial

app = Flask(__name__)
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)

# ===========================
# MOTOR PINS
# ===========================
IN1 = 17
IN2 = 18
IN3 = 22
IN4 = 23

GPIO.setup(IN1, GPIO.OUT)
GPIO.setup(IN2, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT)
GPIO.setup(IN4, GPIO.OUT)

# ===========================
# IR OBSTACLE SENSOR
# ===========================
IR_PIN = 16
GPIO.setup(IR_PIN, GPIO.IN)

def obstacle_detected():
    return GPIO.input(IR_PIN) == 0   # LOW → object close


# ===========================
# SERVO PINS
# ===========================
MAIN_SERVO_PIN = 12
SECOND_SERVO_PIN = 13
ALARM_PIN = 24

pi = pigpio.pi()
pi.set_mode(MAIN_SERVO_PIN, pigpio.OUTPUT)
pi.set_mode(SECOND_SERVO_PIN, pigpio.OUTPUT)

def set_servo(pin, angle):
    if angle < 0: angle = 0
    if angle > 180: angle = 180
    pulse = 500 + (angle * 2000 / 180)
    pi.set_servo_pulsewidth(pin, pulse)

set_servo(MAIN_SERVO_PIN, 90)
set_servo(SECOND_SERVO_PIN, 90)

GPIO.setup(ALARM_PIN, GPIO.OUT)
GPIO.output(ALARM_PIN, 0)

# ===========================
# GLOBAL STATE
# ===========================
x_pos = 0.0
y_pos = 0.0
total_angle = 0.0

DISTANCE_CALIBRATION = 26.0        # cm/sec
LEFT_TURN_CALIBRATION = 80.5           # deg/sec
RIGHT_TURN_CALIBRATION = 87.5
# ===========================
# FLAME SENSOR SERIAL
# ===========================
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=0.1)

def read_flame():
    try:
        line = arduino.readline().decode().strip()
        if line.isdigit():
            return int(line)
    except:
        pass
    return 9999

# ===========================
# MOVEMENT
# ===========================
def stop():
    GPIO.output(IN1, 0)
    GPIO.output(IN2, 0)
    GPIO.output(IN3, 0)
    GPIO.output(IN4, 0)


# =====================================================
# ACCURATE FORWARD WITH OBSTACLE DETECTION
# RETURNS: actual distance traveled (cm)
# =====================================================
def forward_motor(duration):
    GPIO.output(IN1, 1)
    GPIO.output(IN2, 0)
    GPIO.output(IN3, 1)
    GPIO.output(IN4, 0)

    start = time.time()

    while time.time() - start < duration:
        if obstacle_detected():
            print("🟥 OBSTACLE DETECTED — STOPPING")
            stop()

            actual_time = time.time() - start
            actual_dist = actual_time * DISTANCE_CALIBRATION
            return actual_dist

        time.sleep(0.01)

    stop()
    return duration * DISTANCE_CALIBRATION


# =====================================================
# ACCURATE BACKWARD WITH OBSTACLE DETECTION
# (same logic, for safety)
# =====================================================
def backward_motor(duration):
    GPIO.output(IN1, 0)
    GPIO.output(IN2, 1)
    GPIO.output(IN3, 0)
    GPIO.output(IN4, 1)

    start = time.time()

    while time.time() - start < duration:
        if obstacle_detected():
            print("🟥 OBSTACLE BEHIND — STOPPING")
            stop()

            actual_time = time.time() - start
            actual_dist = actual_time * DISTANCE_CALIBRATION
            return actual_dist

        time.sleep(0.01)

    stop()
    return duration * DISTANCE_CALIBRATION


def left_turn_motor(duration):
    GPIO.output(IN1, 0)
    GPIO.output(IN2, 1)
    GPIO.output(IN3, 0)
    GPIO.output(IN4, 0)
    time.sleep(duration)
    stop()

def right_turn_motor(duration):
    GPIO.output(IN1, 0)
    GPIO.output(IN2, 0)
    GPIO.output(IN3, 0)
    GPIO.output(IN4, 1)
    time.sleep(duration)
    stop()

# ===========================
# NON
# ===========================
def async_forward(dist):
    duration = dist / DISTANCE_CALIBRATION
    forward_motor(duration)
def async_backward(dist):
    duration = dist / DISTANCE_CALIBRATION
    backward_motor(duration)
def async_left(angle):
    duration = angle / LEFT_TURN_CALIBRATION
    left_turn_motor(duration)
def async_right(angle):
    duration = angle / RIGHT_TURN_CALIBRATION
    right_turn_motor(duration)
# ===========================
# ANGLE HELPERS
# ===========================
def normalize_angle(a):
    return a % 360

def update_turn(delta):
    global total_angle
    total_angle = normalize_angle(total_angle + delta)

def normalized_dir(angle):
    a = normalize_angle(angle)
    if a < 45 or a >= 315:
        return 0
    if a < 135:
        return 90
    if a < 225:
        return 180
    return 270

def update_forward(dist):
    global x_pos, y_pos, total_angle
    d = normalized_dir(total_angle)
    if d == 0:  x_pos += dist
    elif d == 90: y_pos += dist
    elif d == 180: x_pos -= dist
    else: y_pos -= dist

def update_backward(dist):
    global x_pos, y_pos, total_angle
    d = normalized_dir(total_angle)
    if d == 0:  x_pos -= dist
    elif d == 90: y_pos -= dist
    elif d == 180: x_pos += dist
    else: y_pos += dist

# ===========================
# SECOND SERVO SWEEP
# ===========================
def secondary_servo_sweep(center_angle, duration_seconds=20):
    start = time.time()
    while time.time() < start + duration_seconds:
        for ang in range(center_angle - 45, center_angle + 46, 3):
            set_servo(SECOND_SERVO_PIN, max(0, min(180, ang)))
            time.sleep(0.02)
        for ang in range(center_angle + 45, center_angle - 46, -3):
            set_servo(SECOND_SERVO_PIN, max(0, min(180, ang)))
            time.sleep(0.02)
    set_servo(SECOND_SERVO_PIN, 90)

# ===========================
# FIRE SEARCH (untouched)
# ===========================
def search_fire():
    global x_pos, y_pos, total_angle

    start = time.time()

    while time.time() - start < 600:

        arduino.reset_input_buffer()
        set_servo(MAIN_SERVO_PIN, 90)
        time.sleep(0.1)

        min_value = 999999
        min_servo_angle = 90

        for ang in range(45, 136):
            set_servo(MAIN_SERVO_PIN, ang)
            time.sleep(0.03)

            value = read_flame()
            print("[SCAN]", ang, value)

            if value < min_value:
                min_value = value
                min_servo_angle = ang

            if value < 10:
                stop()
                print("🔥 FIRE AT", ang, value)

                set_servo(MAIN_SERVO_PIN, ang)
                GPIO.output(ALARM_PIN, 1)

                secondary_servo_sweep(ang, 20)

                GPIO.output(ALARM_PIN, 0)
                set_servo(MAIN_SERVO_PIN, 90)
                set_servo(SECOND_SERVO_PIN, 90)

                return {"fire": True}

        offset = min_servo_angle - 90
        if offset > 0:
                duration = abs(offset) / LEFT_TURN_CALIBRATION
                left_turn_motor(duration)
                update_turn(offset)
        elif offset < 0:
                duration = abs(offset) / RIGHT_TURN_CALIBRATION
                right_turn_motor(duration)
                update_turn(offset)
        else:
                pass

        dist = 50
        actual = forward_motor(dist / DISTANCE_CALIBRATION)
        update_forward(actual)

    return {"fire": False}

# ===========================
# ROUTES (UPDATED FOR ACCURATE DIST)
# ===========================
@app.route("/forward")
def handle_forward():
    dist = float(request.args.get("distance", 10))
    threading.Thread(target=async_forward, args=(dist,), daemon=True).start()
    return {"status": "ok"}

@app.route("/backward")
def handle_backward():
    dist = float(request.args.get("distance", 10))
    threading.Thread(target=async_backward, args=(dist,), daemon=True).start()
    return {"status": "ok"}

@app.route("/left")
def handle_left():
    ang = float(request.args.get("angle", 90))
    threading.Thread(target=async_left, args=(ang,), daemon=True).start()
    return {"status": "ok"}

@app.route("/right")
def handle_right():
    ang = float(request.args.get("angle", 90))
    threading.Thread(target=async_right, args=(ang,), daemon=True).start()
    return {"status" : "ok"}

@app.route("/fire")
def handle_fire():
    threading.Thread(target=search_fire, daemon=True).start()
    return {"status" : "fire_sequence_started"}

@app.route("/status")
def status():
    return {"x": x_pos, "y": y_pos, "angle": total_angle}

@app.route("/cometobase")
def cometobase():
    global x_pos, y_pos, total_angle

    dx = -x_pos
    dy = -y_pos

    if abs(dx) < 1 and abs(dy) < 1:
        return {"status": "already_at_base"}

    target_angle = math.degrees(math.atan2(dy, dx)) % 360
    delta = (target_angle - total_angle + 180) % 360 - 180

    if delta > 0:
        right_turn_motor(delta / RIGHT_TURN_CALIBRATION)
    else:
        left_turn_motor(abs(delta) / LEFT_TURN_CALIBRATION)

    update_turn(delta)

    dist = math.sqrt(dx*2 + dy*2)
    actual = forward_motor(dist / DISTANCE_CALIBRATION)
    update_forward(actual)

    return {
        "status": "returned_to_base",
        "final_x": x_pos,
        "final_y": y_pos,
        "final_angle": total_angle
    }

# ===========================
# RUN
# ===========================
if __name__ == "__main__":
    set_servo(12,90)
    app.run(host="0.0.0.0", port=5000, threaded=True)