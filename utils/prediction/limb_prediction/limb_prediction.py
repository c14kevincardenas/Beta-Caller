import torch


def predict_limb(frames, prev_limbs, models, device, use_seq):
    (detr_model, detr_processor), (beit_model, beit_processor), (climbeit_model, climbeit_processor) = models
    inputs = preprocess(frames, detr_processor, detr_model, beit_processor, climbeit_processor, device, use_seq)
    if inputs:
        if use_seq:
            prev_limbs = torch.tensor(prev_limbs)
            return sequence_inference(inputs, prev_limbs, climbeit_model)
        else:  # single frame inference
            return single_frame_inference(inputs, beit_model)
    return None


if __name__ == "__main__":
    from models import load_limb_models
    from preprocessing import preprocess
    from inference import single_frame_inference, sequence_inference
    import cv2
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detr_model_name = "facebook/detr-resnet-50"
    beit_model_name = "c14kevincardenas/ClimBEiT_single_frame"
    climbeit_model_name = "c14kevincardenas/ClimBEiTv2"
    id2label = {0: "left_hand", 1: "right_hand", 2: "left_foot", 3: "right_foot"}

    models = load_limb_models(detr_model_name, beit_model_name, climbeit_model_name, device)

    # example for very first image
    img_path = "C:/Users/Student/Desktop/beta_caller/training_images/boulder/03-02.jpg"
    frames = [cv2.imread(img_path)]
    prev_limbs = []
    print('\n=== Start First Frame Inference ===')
    start_time = time.perf_counter()
    pred = predict_limb(frames, prev_limbs, models, device, use_seq=False)
    total_time = time.perf_counter() - start_time
    print(f"First frame prediction: {id2label[pred]} | Time = {total_time:.4f} seconds")

    # example for second image with only one previous limb
    seq_paths = ["C:/Users/Student/Desktop/beta_caller/training_images/boulder/03-02.jpg",
                 "C:/Users/Student/Desktop/beta_caller/training_images/boulder/03-03.jpg"]
    frames = [cv2.imread(img_path) for img_path in seq_paths]
    prev_limbs = [0]
    print('\n=== Start Second Frame Inference ===')
    start_time = time.perf_counter()
    pred = predict_limb(frames, prev_limbs, models, device, use_seq=False)
    total_time = time.perf_counter() - start_time
    print(f"Second frame prediction: {id2label[pred]} | Time = {total_time:.4f} seconds")

    # example for full sequence (2 frames, 2 prev limbs)
    seq_paths = ["C:/Users/Student/Desktop/beta_caller/training_images/boulder/03-03.jpg",
                 "C:/Users/Student/Desktop/beta_caller/training_images/boulder/03-04.jpg"]
    frames = [cv2.imread(img_path) for img_path in seq_paths]
    prev_limbs = [0, 3]
    print('\n=== Start Sequence Inference ===')
    start_time = time.perf_counter()
    pred = predict_limb(frames, prev_limbs, models, device, use_seq=True)
    total_time = time.perf_counter() - start_time
    print(f"Third frame (sequence) prediction: {id2label[pred]} | Time = {total_time:.4f} seconds")

else:
    from utils.prediction.limb_prediction.preprocessing import preprocess
    from utils.prediction.limb_prediction.inference import single_frame_inference, sequence_inference