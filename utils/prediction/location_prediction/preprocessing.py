import torch
import cv2


def preprocess(frame, processor):
    return processor(images=frame, return_tensors="pt")


if __name__ == "__main__":
    from models import load_location_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_model_name = 'c14kevincardenas/ClimBEiT_single_frame'
    hands_model_name = 'c14kevincardenas/limbxy_hands'
    feet_model_name = 'c14kevincardenas/limbxy_feet'
    print('Loading Limb Prediction Models...')
    (hands_model, hands_processor), (feet_model, feet_processor) = load_location_models(img_model_name,
                                                                                        hands_model_name,
                                                                                        feet_model_name,
                                                                                        device)

    # example usage
    img_path = "C:/Users/Student/Desktop/beta_caller/training_images/boulder/03-03.jpg"
    frame = cv2.imread(img_path)
    preprocessed_hands_data = preprocess(frame, hands_processor)
    preprocessed_feet_data = preprocess(frame, feet_processor)

    print("\n=== Preprocessing Hands Results ===")
    print(f"Preprocessed Data Keys: {preprocessed_hands_data.keys()}")
    print(f"Pixel Values Shape: {preprocessed_hands_data['pixel_values'].shape}")

    print("\n=== Preprocessing Feet Results ===")
    print(f"Preprocessed Data Keys: {preprocessed_feet_data.keys()}")
    print(f"Pixel Values Shape: {preprocessed_feet_data['pixel_values'].shape}")
