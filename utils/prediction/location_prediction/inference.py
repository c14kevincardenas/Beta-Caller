import torch


def inference(img_inputs, prev_limb, model):
    with torch.no_grad():
        pred = model(pixel_values=img_inputs["pixel_values"], limbs=prev_limb)
    return pred[0]
