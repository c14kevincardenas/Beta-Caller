from PIL import Image
import torch
import cv2


def crop_person(frame, detr_processor, model, device):
    inputs = detr_processor(images=frame, return_tensors="pt").to(device)

    # Perform object detection
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([frame.shape[:2]])
    results = detr_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.85)[0]
    person_boxes = [
        box for box, score, label in zip(results["boxes"], results["scores"], results["labels"])
        if label == 1 and score > 0.9
    ]

    if person_boxes:
        box = person_boxes[0].cpu().numpy().astype(int)
        cropped_image = frame[box[1]:box[3], box[0]:box[2]]
        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    else:
        print("No person detected")
        return None


def preprocess(frames, detr_processor, detr_model, beit_processor, climbeit_processor, device, use_seq):
    cropped_images = []
    for frame in frames:
        crop_image = crop_person(frame, detr_processor, detr_model, device)
        if crop_image:
            cropped_images.append(crop_image.resize((384, 384)))

    if cropped_images:
        if use_seq:
            return climbeit_processor(images=cropped_images, return_tensors="pt")
        else:
            return beit_processor(images=cropped_images[-1], return_tensors="pt")
    return None


if __name__ == "__main__":
    from models import load_limb_models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detr_model_name = "facebook/detr-resnet-50"
    beit_model_name = "c14kevincardenas/ClimBEiT_single_frame"
    climbeit_model_name = "c14kevincardenas/ClimBEiTv2"
    id2label = {0: "left_hand", 1: "right_hand", 2: "left_foot", 3: "right_foot"}

    (detr_model, detr_processor), (beit_model, beit_processor), (climbeit_model, climbeit_processor) = load_limb_models(
        detr_model_name, beit_model_name, climbeit_model_name, device)

    # example usage
    img_path = "C:/Users/Student/Desktop/beta_caller/training_images/boulder/03-03.jpg"
    frame = cv2.imread(img_path)
    cropped_image = crop_person(frame, detr_processor, detr_model, device)
    if cropped_image is not None:
        cropped_image.show()  # Show the cropped image for verification
        preprocessed_data = preprocess(
            frames=[frame],
            detr_processor=detr_processor,
            detr_model=detr_model,
            beit_processor=beit_processor,
            climbeit_processor=None,  # Pass None if not using climbeit_processor
            device=device,
            use_seq=False,
        )

        print("\n=== Preprocessing Results ===")
        print(f"Preprocessed Data Keys: {preprocessed_data.keys()}")
        print(f"Pixel Values Shape: {preprocessed_data['pixel_values'].shape}")

    else:
        print("No person detected in the frame.")
