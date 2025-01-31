import torch


def single_frame_inference(inputs, model):
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = logits.argmax(-1).item()
    return pred


def sequence_inference(img_inputs, prev_limbs, model):
    with torch.no_grad():
        logits = model(pixel_values=img_inputs["pixel_values"], prev_labels=prev_limbs)
        pred = logits.argmax(-1).item()
    return pred
