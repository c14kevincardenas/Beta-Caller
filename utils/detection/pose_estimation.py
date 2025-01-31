import cv2
import numpy as np
from easy_ViTPose import VitInference
from utils.detection.path_manager import VITPOSE_MODEL_PATH, YOLO_MODEL_PATH

print('Loading Pose Estimation Model...')
model = VitInference(VITPOSE_MODEL_PATH, YOLO_MODEL_PATH, model_name='s', yolo_size=320, is_video=False, device=None)


def calculate_ipd(kps, frame):
    """
    Calculate interpupillary distance (IPD) from keypoints.
    Args:
        keypoints (list): List of pose keypoints in the format [(x, y, confidence), ...].
        frame (np.ndarray): The image frame.

    Returns:
        float: Interpupillary distance in pixels.
    """
    left_eye = kps[1]
    right_eye = kps[2]
    h, w, _ = frame.shape

    ipd = np.linalg.norm(right_eye - left_eye)
    return ipd


def process_results(results, frame):
    if not results:
        return np.array([])

    # Select the most confident pose
    most_confident_key = None
    highest_avg_confidence = -1

    for person_id, pose in results.items():
        # Compute the average confidence score
        avg_confidence = np.mean(pose[:, 2])  # Third column is confidence
        if avg_confidence > highest_avg_confidence:
            highest_avg_confidence = avg_confidence
            most_confident_key = person_id

    if most_confident_key is None:
        return np.array([])  # No confident pose found

    lms = results[most_confident_key]  # Get the most confident pose
    h, w, c = frame.shape

    # normalize y and x
    lms[:, 0] /= h
    lms[:, 1] /= w
    norm_lms = np.delete(lms, 2, axis=1)

    # switch from (y, x) to (x, y)
    norm_lms[:, [0, 1]] = norm_lms[:, [1, 0]]

    return norm_lms


def estimate_pose(frame, draw):
    # Convert BGR to RGB for the model
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # run pose estimation
    keypoints = model.inference(rgb_frame)
    if not keypoints:
        # no keypoints detected
        return np.array([]), frame

    # Draw keypoints on the frame
    if keypoints and draw:
        model._keypoints = keypoints
        frame = model.draw(show_yolo=False, confidence_threshold=0.2)

    # get normalized keypoints for the most confident pose
    norm_kps = process_results(keypoints, frame)

    return norm_kps, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    cam_num = 1
    cap = cv2.VideoCapture(cam_num)
    while True:
        success, frame = cap.read()
        if not success:
            continue

        frame, kps = estimate_pose(frame, draw=True)
        cv2.imshow("Beta Caller", cv2.resize(frame, (800, 600)))

        key = cv2.waitKey(1) & 0xFF
        if cv2.getWindowProperty("Beta Caller", cv2.WND_PROP_VISIBLE) < 1 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
