import cv2
import torch
import time
from utils.data.data_manager import PreviousMoves
from utils.detection.hold_detection import detect_motion, detect_holds, draw_hold_bbs
from utils.detection.pose_estimation import estimate_pose, calculate_ipd
from utils.detection.movement_detection import MovementDetector
from utils.prediction.move_prediction import predict_move
from utils.prediction.limb_prediction.models import load_limb_models, load_limb_models_student
from utils.prediction.location_prediction.models import load_location_models
from utils.prediction.command_translation import calculate_direction_and_distance
from utils.prediction.call_prediction import TextToSpeech, call_prediction

movement_detector = MovementDetector(movement_threshold=0.085, stability_threshold=6)
previous_moves = PreviousMoves(save_dir='C:/Users/Student/Desktop/beta_caller/training_images/saved_moves')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detr_model_name = 'facebook/detr-resnet-50'
img_model_name = 'c14kevincardenas/ClimBEiT_single_frame'
img_model_name_stud = 'microsoft/beit-base-patch16-224'
climbeit_limb_model_name = 'c14kevincardenas/ClimBEiT-t2'
# climbeit_limb_model_name = 'c14kevincardenas/ClimBEiT-student-t2'
hands_loc_model_name = 'c14kevincardenas/limbxy_hands'
feet_loc_model_name = 'c14kevincardenas/limbxy_feet'
if 'student' in climbeit_limb_model_name.split('-'):
    print('Loading ClimBEiT Student Limb Prediction Model...')
    limb_models = load_limb_models_student(detr_model_name, img_model_name, img_model_name_stud, climbeit_limb_model_name, device)
else:
    print('Loading ClimBEiT Limb Prediction Model...')
    limb_models = load_limb_models(detr_model_name, img_model_name, climbeit_limb_model_name, device)
print('Loading Limb Location Models...')
location_models = load_location_models(img_model_name, hands_loc_model_name, feet_loc_model_name, device)
tts = TextToSpeech(rate=140, voice_index=1)


def main(cam_num, draw_holds, draw_pose):
    # start video capture
    cap = cv2.VideoCapture(cam_num)
    _, prev_frame = cap.read()
    prev_frame = cv2.resize(prev_frame, (320, 240))  # Resize for faster processing

    # initialize variables and flags
    holds = None
    ipd_pixels = None
    target_hold = None
    location = None
    run_hold_detection = True
    ipd_calc_mode = False
    run_beta_caller = False
    use_seq = False

    print("Adjust camera. 'd': done with finding holds.")
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
                print("Motion detected. Running hold detection...")
                holds, frame = detect_holds(frame)
                prev_frame = resized_frame  # Update the previous frame
            else:
                frame = draw_hold_bbs(frame, holds)

        # wait for user to calculate interpupilary distance when climber is at the wall facing the camera
        elif ipd_calc_mode:
            # run pose estimation
            kps, pose_frame = estimate_pose(frame, draw=draw_pose)
            if len(kps) == 0:
                continue

            # interpupilary distance
            ipd_pixels = calculate_ipd(kps, frame)

            # check to continue drawing pose and/or holds on the wall
            if draw_pose:
                frame = pose_frame
            if draw_holds:
                frame = draw_hold_bbs(frame, holds)

        # run beta caller
        elif run_beta_caller:
            # run pose estimation
            kps, pose_frame = estimate_pose(frame, draw=draw_pose)
            if len(kps) == 0:
                continue

            # update movement detector
            state = movement_detector.update(kps)

            # check if the climber completed a move to predict, process, and transmit the next command
            if state == "move_completed":
                print("\nMove completed!")
                # check if more than two moves have been collected (can then use sequence model)
                if len(previous_moves.recent_limbs) >= 2:
                    use_seq = True

                # make move prediction
                start_time = time.perf_counter()
                limb, location = predict_move(frame, previous_moves, kps, limb_models, location_models, device, use_seq)
                total_time = time.perf_counter() - start_time
                print(f"\tTotal Move Prediction Time = {total_time:.4f} seconds")

                # translate location prediction into direction and distance from location and keypoints
                direction, distance, target_hold = calculate_direction_and_distance(limb, kps, holds, location,
                                                                                    ipd_pixels)

                # call move (transmit command to climber)
                call_prediction(tts, limb, direction, distance)

            elif state == "moving":
                print("Climber is moving...")
            elif state == "stopped":
                print("Climber is stable, no movement detected.")

            # check to continue drawing pose and/or holds on the wall
            if draw_pose:
                frame = pose_frame
            if draw_holds:
                frame = draw_hold_bbs(frame, holds, target_hold, location)

        # display image
        cv2.imshow("Beta Caller", cv2.resize(frame, (800, 600)))

        # handle key events
        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            run_hold_detection = not run_hold_detection  # toggle hold detection mode
            print("Hold detection mode:", "ON" if run_hold_detection else "OFF")
            ipd_calc_mode = not ipd_calc_mode  # toggle ipd calc mode
            print("Calculate interpupillary distance mode:", "ON" if ipd_calc_mode else "OFF")
            if ipd_calc_mode:
                print("\nClimber, move to the wall and face the camera.\n'd': back to finding holds. 'h': draw holds. "
                      "'p': draw pose. 'ENTER': start Beta Caller.")
        elif key in [10, 13] and len(holds) > 0 and ipd_pixels is not None:  # pressed ENTER to start Beta Caller
            run_hold_detection = False
            ipd_calc_mode = False
            run_beta_caller = True
            print('\nCLIMB ON!!')
        elif cv2.getWindowProperty("Beta Caller", cv2.WND_PROP_VISIBLE) < 1 or key == ord('q'):
            break
        elif key == ord('h'):
            draw_holds = not draw_holds  # toggle drawing holds
            print("Drawing Holds:", "ON" if draw_holds else "OFF")
        elif key == ord('p'):
            draw_pose = not draw_pose  # toggle drawing climber's pose
            print("Drawing Pose:", "ON" if draw_pose else "OFF")

    # turn off camera and close window
    cap.release()
    cv2.destroyAllWindows()

    # turn of text to speech engine
    tts.stop()


if __name__ == '__main__':
    draw_holds = True
    draw_pose = True
    cam_num = 1
    main(cam_num, draw_holds, draw_pose)
