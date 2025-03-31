import numpy as np


def predict_location_dnn(kps, limb, location_models):
    _, _, model = location_models
    inputs = np.array([np.append(kps.flatten(), limb), ])
    location = model.predict(inputs)[0]
    return location


if __name__ == '__main__':
    import os
    import sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    print(parent_dir)
    sys.path.insert(0, parent_dir)
    from utils.detection.pose_estimation import estimate_pose
    from tensorflow.keras.models import load_model
    import cv2
    import numpy as np

    # load model and image
    model = load_model(parent_dir + '/models/limb_xy.keras')
    img_path = "C:/Users/Student/Desktop/beta_caller/training_images/boulder/05-09.jpg"
    image = cv2.imread(img_path)
    new_size = (400, 600)
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    # get keypoints
    kps, pose_frame = estimate_pose(image, draw=True)
    limb = 0

    # make inputs
    inputs = np.array([np.append(kps.flatten(), limb), ])
    print(inputs)
    print(inputs.shape)

    # make limb x,y location prediction
    location = model.predict(inputs)[0]
    print(location)

    # draw prediction
    h, w, c = image.shape
    x, y = int(location[0] * w), int(location[1] * h)
    print(x, y)
    radius = 10
    color = (0, 100, 255)
    thickness = -1
    cv2.circle(image, (x, y), radius, color, thickness)

    # show the image
    cv2.imshow("Image with Circle", image)
    cv2.waitKey(0)  # wait for a key press
    cv2.destroyAllWindows()  # close the window
