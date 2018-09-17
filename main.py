import numpy as np
import cv2
import imutils
import datetime
import subprocess
import os

# Advance top view by this time (milliseconds)
TOP_DELAY_TIME = 1500

# Sync trigger when the frame difference is greater than correction
CORRECTION_SENSITIVITY = 0.1

# Duration of "Red" Stop light
MAX_STOP_TIME = 10

# Duration of "Green" Stop light
MAX_GO_TIME = 5

# Set initial value for stop light (True for "Red")
INTIALLY_STOPPED = True

# Detection reset timer
TRIGGER_CD = 1

# Object detection Region
ROI_X_MIN = 400
ROI_X_MAX = 1728
ROI_Y_MIN = 50
ROI_Y_MAX = 800

# Trigger height
ROI_TRIGGER_Y = 384

# Video file for top view
VIDEO_SOURCE_TOP = "ztop24.mov"

# Video file for front view
VIDEO_SOURCE_FRONT = "zfront24.mp4"

# Reference image for background subtraction
IMAGE_REF_BACKGROUND = "new-ref.jpg"

# output folder
OUTPUT_DIR = "raw"

# crop output (default)
# OUT_X_MIN = 0
# OUT_X_MAX = 1920
# OUT_Y_MIN = 0
# OUT_Y_MAX = 1080

OUT_X_MIN = 0
OUT_X_MAX = 1920
OUT_Y_MIN = 500
OUT_Y_MAX = 1000

# Preview size
FRONT_VIEW_WIDTH = 500
TOP_VIEW_WIDTH = 500

# Minimum Obstruction Object Area
MIN_OBS_AREA = 2800
# Maximum Obsctruction Object Area
MAX_OBS_AREA = 30000

if not os.path.exists(OUTPUT_DIR + "/top"):
    os.makedirs(OUTPUT_DIR + "/top")

if not os.path.exists(OUTPUT_DIR + "/front"):
    os.makedirs(OUTPUT_DIR + "/front")

cap = cv2.VideoCapture(VIDEO_SOURCE_TOP)
cap2 = cv2.VideoCapture(VIDEO_SOURCE_FRONT)

ref = cv2.imread(IMAGE_REF_BACKGROUND)

gray_ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
gray_ref = cv2.GaussianBlur(gray_ref, (21, 21), 0)

time = 0
stopped = True
if stopped:
    print("[STOP]")
else:
    print("[GO]")
stop_time = 0
go_time = 0

trigger_time = 0
trigger_ready = True

detect_count = 0

fps1 = cap.get(cv2.CAP_PROP_FPS)
fps2 = cap2.get(cv2.CAP_PROP_FPS)
frame1_count = 0
frame2_count = 0
cap.get(cv2.CAP_PROP_FRAME_COUNT)
while (cap.isOpened() and cap2.isOpened()):
    ret, frame = cap.read()
    frame1_count += 1

    raw_time = cap.get(cv2.CAP_PROP_POS_MSEC)
    if raw_time < TOP_DELAY_TIME:
        frame1_count = 0
        continue
    t = int(raw_time / 1000)

    ret2, frame2 = cap2.read()
    frame2_count += 1

    t1 = frame1_count / fps1
    t2 = frame2_count / fps2

    while (t2 - t1) > CORRECTION_SENSITIVITY:
        ret, frame = cap.read()
        frame1_count += 1
        t1 = frame1_count / fps2
    while (t1 - t2) > CORRECTION_SENSITIVITY:
        ret2, frame2 = cap2.read()
        frame2_count += 1
        t2 = frame2_count / fps2

    # This line flips the video, uncomment if necessary
    # frame = cv2.flip(frame, -1)
    frame2 = cv2.flip(frame2, -1)

    if t > time:
        if trigger_time >= TRIGGER_CD:
            trigger_ready = True
        else:
            trigger_time += 1
        time = t
        if stopped:
            stop_time += 1
            if stop_time >= MAX_STOP_TIME:
                stopped = False
                stop_time = 0
                print("[GO]")
        else:
            go_time += 1
            if go_time >= MAX_GO_TIME:
                stopped = True
                go_time = 0
                print("[STOP]")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    frameDelta = cv2.absdiff(gray_ref, gray)
    thresh = cv2.threshold(frameDelta, 100, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    for c in cnts:
        if cv2.contourArea(c) < MIN_OBS_AREA or cv2.contourArea(
                c) > MAX_OBS_AREA:
            continue

        (x, y, w, h) = cv2.boundingRect(c)

        if x < ROI_X_MIN or x > ROI_X_MAX or y < ROI_Y_MIN or y > ROI_Y_MAX:
            continue

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if stopped and trigger_ready and ROI_TRIGGER_Y > (y +
                                                          (h / 2)) > ROI_Y_MIN:
            print("{} : TRIGGER DETECTED! [{}]".format(
                datetime.timedelta(milliseconds=raw_time), detect_count))
            front_snapshot = '{}/front/{}-{}.jpg'.format(
                OUTPUT_DIR,
                VIDEO_SOURCE_FRONT.split(".")[0], detect_count)
            cv2.imwrite(front_snapshot,
                        frame2[OUT_Y_MIN:OUT_Y_MAX, OUT_X_MIN:OUT_X_MAX])
            top_snapshot = '{}/top/{}-{}.jpg'.format(
                OUTPUT_DIR,
                VIDEO_SOURCE_TOP.split(".")[0], detect_count)
            cv2.imwrite(top_snapshot, frame)

            subprocess.Popen([
                "python", "detector.py", VIDEO_SOURCE_TOP, VIDEO_SOURCE_FRONT,
                str(datetime.timedelta(milliseconds=raw_time)), top_snapshot,
                front_snapshot
            ])

            detect_count += 1
            trigger_time = 0
            trigger_ready = False

    cv2.rectangle(frame, (ROI_X_MIN, ROI_Y_MIN), (ROI_X_MAX, ROI_TRIGGER_Y),
                  (0, 0, 255), 2)
    cv2.rectangle(frame, (ROI_X_MIN, ROI_Y_MIN), (ROI_X_MAX, ROI_Y_MAX),
                  (255, 0, 0), 2)

    cv2.rectangle(frame2, (OUT_X_MIN, OUT_Y_MIN), (OUT_X_MAX, OUT_Y_MAX),
                  (255, 0, 0), 2)

    frame = imutils.resize(frame, width=TOP_VIEW_WIDTH)
    frame2 = imutils.resize(frame2, width=FRONT_VIEW_WIDTH)

    cv2.imshow('TOP VIEW', frame)
    cv2.imshow('FRONT VIEW', frame2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
