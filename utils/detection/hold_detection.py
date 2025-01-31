import cv2
import base64
import numpy as np
import requests

ROBOFLOW_API_KEY = 'VJMD3QMBnozquVzWmrQd'
ROBOFLOW_MODEL = 'hold_detection_v3'
ROBOFLOW_VERSION_NUMBER = '5'
ROBOFLOW_SIZE = 640
ROBOFLOW_CONFIDENCE = 40
ROBOFLOW_OVERLAP = 60

upload_url = ''.join([
    'https://detect.roboflow.com/',
    ROBOFLOW_MODEL, '/',
    ROBOFLOW_VERSION_NUMBER,
    '?api_key=',
    ROBOFLOW_API_KEY,
    '&format=json',
    '&stroke=5',
    f'&confidence={ROBOFLOW_CONFIDENCE}',
    f'&overlap={ROBOFLOW_OVERLAP}'
])


# motion detection function
def detect_motion(prev_frame, curr_frame, threshold=5000):
    diff = cv2.absdiff(prev_frame, curr_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    motion_score = cv2.countNonZero(thresh)
    return motion_score > threshold  # Return True if significant motion is detected


def detect_holds(frame, draw=True):
    holds = np.empty((0, 4), int)
    # scores = np.empty(0, float)

    # resize frame for Roboflow API
    h, w, c = frame.shape
    scale = ROBOFLOW_SIZE / max(h, w)
    resized_frame = cv2.resize(frame, (round(scale * w), round(scale * h)))

    # encode image to base64 string
    retval, buffer = cv2.imencode('.jpg', resized_frame)
    img_str = base64.b64encode(buffer)

    # call Roboflow API
    resp = requests.post(upload_url, data=img_str, headers={'Content-Type': 'application/x-www-form-urlencoded'})
    predictions = resp.json().get('predictions', [])

    for hold in predictions:
        xmin = int(hold['x'] - hold['width'] / 2)
        ymin = int(hold['y'] - hold['height'] / 2)
        xmax = int(hold['x'] + hold['width'] / 2)
        ymax = int(hold['y'] + hold['height'] / 2)

        # rescale bounding box coordinates back to the original frame size
        xmin = int(xmin / scale)
        ymin = int(ymin / scale)
        xmax = int(xmax / scale)
        ymax = int(ymax / scale)

        # draw bounding boxes
        if draw:
            frame = cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (20, 255, 57), 2)

        # append data
        holds = np.append(holds, np.array([[xmin / w, ymin / h, xmax / w, ymax / h]]), axis=0)
        # scores = np.append(scores, hold['confidence'])

    return holds, frame


def draw_hold_bbs(frame, holds, target_hold=None):
    if holds is not None:
        h, w, _ = frame.shape
        for xmin, ymin, xmax, ymax in holds:
            frame = cv2.rectangle(
                frame,
                (int(xmin * w), int(ymin * h)),
                (int(xmax * w), int(ymax * h)),
                (20, 255, 57),
                thickness=2
            )
    if target_hold is not None:
        h, w, _ = frame.shape
        xmin, ymin, xmax, ymax = target_hold
        frame = cv2.rectangle(
            frame,
            (int(xmin * w), int(ymin * h)),
            (int(xmax * w), int(ymax * h)),
            (0, 0, 255),
            thickness=3
        )

    return frame


if __name__ == '__main__':
    draw_holds = True
    cam_num = 1
    cap = cv2.VideoCapture(cam_num)
    _, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (320, 240))  # Resize for faster processing

    # initialize variables and flags
    holds = None
    run_hold_detection = True

    print('Adjust the camera to include all holds. Press "d" when done. Press "q" to quit or close the screen.')
    # run hold detection while the user has not quit or is not done adjusting the camera
    while True:
        # read the current frame
        success, frame = cap.read()
        if not success:
            continue

        # resize frame for motion detection
        resized_frame = cv2.resize(frame, (320, 240))

        # run hold detection
        if run_hold_detection:
            if detect_motion(prev_frame, resized_frame):
                print('Motion detected. Running hold detection...')
                holds, frame = detect_holds(frame)
                prev_frame = resized_frame  # Update the previous frame
            else:
                frame = draw_hold_bbs(frame, holds)

        # run Beta Caller
        else:
            if draw_holds:
                frame = draw_hold_bbs(frame, holds)

        # display image
        cv2.imshow('Beta Caller', cv2.resize(frame, (800, 600)))

        # handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            run_hold_detection = not run_hold_detection  # Toggle detection mode
            print('Hold detection mode:', 'ON' if run_hold_detection else 'OFF')
        elif cv2.getWindowProperty('Beta Caller', cv2.WND_PROP_VISIBLE) < 1 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
