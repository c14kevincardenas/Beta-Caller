from utils.prediction.limb_prediction.limb_prediction import predict_limb
from utils.prediction.location_prediction.location_prediction import predict_location
from utils.prediction.location_prediction.location_prediction_dnn import predict_location_dnn


def predict_move(frame, previous_moves, kps, limb_models, location_models, device, use_seq):
    """
    Handles data storage and predictions when a climber completes a move.

    Args:
        frame (np.ndarray): The current video frame.
        previous_moves (PreviousMoves): Instance of the data manager class.
    """
    # if 2 moves are already stored, pop and save oldest frame to disk
    if use_seq:
        previous_moves.pop_and_save()
        # print(f'\tPopped and saved frame! # Recent Frames: {len(previous_moves.recent_frames)}')

    # add current frame to recent data to use for sequence-based inference
    previous_moves.add_frame(frame)
    # print(f'\tAdded frame! # Recent Frames: {len(previous_moves.recent_frames)}')

    # predict next limb using ClimBEiT
    frames, prev_limbs = previous_moves.get_recent_data()
    print(f'\tPredicting Limb! Frames: {len(frames)} | Limbs: {prev_limbs}')
    limb = predict_limb(frames, prev_limbs, limb_models, device, use_seq)
    # limb = 1  # placeholder for limb prediction

    # add predicted limb to recent data
    previous_moves.add_limb(limb)
    # print(f'\tAdded limb! {previous_moves.recent_limbs}')

    # predict next location using ClimBEiT
    prev_limbs = previous_moves.recent_limbs
    print(f'\tPredicting Location! Frames: {len(frames)} | Limbs: {prev_limbs}')
    if len(kps) > 0:
        location = predict_location_dnn(kps, limb, location_models)
    else:  # kps not identified in frame, use ClimBEiT
        print('No keypoints, using ClimBEiT!')
        location = predict_location(frames, prev_limbs, location_models)
    # location = (0.5, 0.5)  # placeholder for location prediction

    return limb, location
