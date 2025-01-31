import torch


def predict_location(frames, prev_limbs, models):
    frame = frames[-1]
    last_limb = prev_limbs[-1]
    (hands_model, hands_processor), (feet_model, feet_processor) = models

    # check which model and processor to use
    if last_limb in [0, 1]:
        model, processor = hands_model, hands_processor
    else:  # feet
        model, processor = feet_model, feet_processor

    # get inputs and run inference
    inputs = preprocess(frame, processor)
    if inputs:
        prev_limb = torch.tensor([last_limb])
        return inference(inputs, prev_limb, model)
    return None


if __name__ == "__main__":
    from models import load_location_models
    from preprocessing import preprocess
    from inference import inference
    import cv2
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_model_name = 'c14kevincardenas/ClimBEiT_single_frame'
    hands_model_name = 'c14kevincardenas/limbxy_hands'
    feet_model_name = 'c14kevincardenas/limbxy_feet'
    print('Loading Limb Prediction Models...')
    models = load_location_models(img_model_name, hands_model_name, feet_model_name, device)

    # # example for very first image
    # img_path = "C:/Users/Student/Desktop/beta_caller/training_images/boulder/03-01.jpg"
    # frame = cv2.imread(img_path)
    # prev_limb = []
    # print('\n=== Start First Frame Inference ===')
    # start_time = time.perf_counter()
    # pred = predict_location(frame, prev_limb, model)
    # total_time = time.perf_counter() - start_time
    # print(f"First frame prediction: {pred} | Time = {total_time:.4f} seconds")

    # example for second image with only one previous limb
    seq_paths = ["C:/Users/Student/Desktop/beta_caller/training_images/boulder/03-01.jpg",
                 "C:/Users/Student/Desktop/beta_caller/training_images/boulder/03-02.jpg"]
    frames = [cv2.imread(img_path) for img_path in seq_paths]
    prev_limbs = [1]

    print('\n=== Start Second Frame Inference ===')
    start_time = time.perf_counter()
    pred = predict_location(frames, prev_limbs, models)
    total_time = time.perf_counter() - start_time
    print(f"Second frame prediction: {pred} | Time = {total_time:.4f} seconds")

else:
    from utils.prediction.location_prediction.preprocessing import preprocess
    from utils.prediction.location_prediction.inference import inference
