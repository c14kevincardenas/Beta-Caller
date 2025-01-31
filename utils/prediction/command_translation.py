import numpy as np

AVG_PUPIL_DIST = 2.5


def pixel_to_physical_dist(pixel_distance, ipd_pixels):
    """
    Converts pixel distance to physical distance in inches.
    Args:
        pixel_distance (float): Distance between the climber's moving limb to the target hold in pixels.
        ipd_pixels (float): Interpupillary distance in normalized pixels.

    Returns:
        int: Physical distance between the climber's moving limb to the target hold rounded to the nearest inch.
    """
    conversion_factor = AVG_PUPIL_DIST / ipd_pixels
    physical_distance = pixel_distance * conversion_factor
    # print(pixel_distance, ipd_pixels, conversion_factor, physical_distance)
    return round(physical_distance)


def calculate_direction_and_distance(limb, keypoints, holds, predicted_location, ipd_pixels):
    """
    Calculate the direction (as a clock hour) and distance (in inches) to the target hold based on the predicted limb.

    Args:
        limb (int): Predicted limb (0: left hand, 1: right hand, 2: left foot, 3: right foot).
        keypoints (np.ndarray): Array of keypoints in the format [(x1, y1), (x2, y2), ...].
        holds (np.ndarray): Array of holds in the format [(x_min, y_min, x_max, y_max), ...].
        predicted_location (np.ndarray): Array of x, y coordinate for the predicted limb location for the next move.
        ipd_pixels (float): Interpupillary distance in normalized pixels.

    Returns:
        tuple: (clock_hour, distance, target_hold) where:
            - clock_hour is the hour hand on a clock (1-12).
            - distance is the Euclidean distance from the limb to the target hold.
            - target_hold is the hold the climber should move to in the format (x_min, y_min, x_max, y_max).
    """
    # determine the extremity coordinates
    if limb == 0:  # Left hand
        limb_xy = keypoints[9]  # Left wrist
    elif limb == 1:  # Right hand
        limb_xy = keypoints[10]  # Right wrist
    elif limb == 2:  # Left foot
        limb_xy = keypoints[15]  # Left ankle
    elif limb == 3:  # Right foot
        limb_xy = keypoints[16]  # Right ankle
    else:
        raise ValueError("Invalid limb index. Must be 0, 1, 2, or 3.")

    # extract nose (for hand movement) and hip midpoint (for foot movement) keypoints
    nose_xy = keypoints[0]  # Assuming the first keypoint is the nose
    hip_midpoint_xy = (keypoints[11] + keypoints[12]) / 2  # Midpoint of left and right hips

    # find candidate holds
    def find_candidate_holds(predicted_location, holds, buffer=0.00):
        x, y = predicted_location
        expanded_holds = holds + np.array([-buffer, -buffer, buffer, buffer])
        is_within = (x >= expanded_holds[:, 0]) & (x <= expanded_holds[:, 2]) & \
                    (y >= expanded_holds[:, 1]) & (y <= expanded_holds[:, 3])
        return holds[is_within]

    buffer = 0.00
    candidate_holds = find_candidate_holds(predicted_location, holds)

    while len(candidate_holds) == 0:  # expand buffer until at least one hold is found
        buffer += 0.005
        candidate_holds = find_candidate_holds(predicted_location, holds, buffer)

    # find the center of candidate holds
    hold_centers = np.column_stack((
        (candidate_holds[:, 0] + candidate_holds[:, 2]) / 2,
        (candidate_holds[:, 1] + candidate_holds[:, 3]) / 2
    ))

    # find the closest hold
    distances = np.linalg.norm(hold_centers - limb_xy, axis=1)
    closest_idx = np.argmin(distances)
    target_hold = candidate_holds[closest_idx]
    target_hold_center = hold_centers[closest_idx]
    norm_pixel_distance = distances[closest_idx]

    # convert normalized pixel distance to physical distance using interpupillary distance (between eyes)
    physical_distance = pixel_to_physical_dist(norm_pixel_distance, ipd_pixels)

    # calculate direction based on the limb type
    if limb in [0, 1]:  # hand
        vector = target_hold_center - nose_xy
    else:  # foot
        vector = target_hold_center - hip_midpoint_xy

    angle = np.arctan2(vector[1], vector[0])  # angle in radians
    angle_degrees = (np.degrees(angle) + 90) % 360  # normalize to [0, 360) and add 90 to rotate axis so 0 is up
    clock_hour = int((angle_degrees % 360) // 30) or 12  # convert to clock hour

    return clock_hour, physical_distance, target_hold


if __name__ == '__main__':
    import cv2
    from beta_caller.beta_caller.run_beta_caller.utils.detection.hold_detection import detect_holds
    from beta_caller.beta_caller.run_beta_caller.utils.detection.pose_estimation import estimate_pose

    image_path = '/training_images/boulder_limb/left_hand/03-02.jpg'
    image = cv2.imread(image_path)

    # detect holds
    holds, image = detect_holds(image, draw=True)

    # extract pose keypoints
    kps, image = estimate_pose(image, draw=True)
    # ipd_pixels = calculate_ipd(kps, image)
    ipd_pixels = 0.02  # this is an estimation ipd for video 03

    # predicted limb and location (example values, replace with real predictions)
    limb = 0
    predicted_location = np.array((0.3, 0.35))  # example predicted location in normalized coordinates

    # calculate direction and distance
    clock_hour, distance, target_hold = calculate_direction_and_distance(limb, kps, holds,
                                                                         predicted_location, ipd_pixels)

    # print the results
    print(f"Direction: {clock_hour} o'clock")
    print(f"Distance: {distance:.2f}")

    # convert normalized coordinates to pixel coordinates
    image_height, image_width, _ = image.shape
    predicted_x_pixel = int(predicted_location[0] * image_width)
    predicted_y_pixel = int(predicted_location[1] * image_height)
    xmin_pixel = int(target_hold[0] * image_width)
    ymin_pixel = int(target_hold[1] * image_height)
    xmax_pixel = int(target_hold[2] * image_width)
    ymax_pixel = int(target_hold[3] * image_height)

    # draw a circle at the predicted x,y location
    radius = 10
    color = (0, 0, 255)
    thickness = -1
    cv2.circle(
        image,
        (predicted_x_pixel, predicted_y_pixel),
        radius,
        color,
        thickness=thickness
    )

    # draw a rectangle (bounding box) for the target hold
    box_thickness = 5
    cv2.rectangle(
        image,
        (xmin_pixel, ymin_pixel),
        (xmax_pixel, ymax_pixel),
        color,
        box_thickness
    )

    # see the results
    cv2.imshow("Example Image", cv2.resize(image, (600, 800)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
