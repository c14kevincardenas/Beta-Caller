import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    AutoModel,
    AutoProcessor,
    AutoConfig
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class LimbXYModel(PreTrainedModel):
    def __init__(self, image_model, config, num_coordinates=2):
        super(LimbXYModel, self).__init__(config)
        # set config
        self.config = config

        # load the pre-trained model
        self.image_model = image_model

        # # freeze the pre-trained layers from image classification
        # for param in self.image_model.parameters():
        #     param.requires_grad = False

        # sequential container with Linear layer followed by sigmoid activation
        self.regression_head = nn.Sequential(
            nn.Linear(self.image_model.config.hidden_size + 8, num_coordinates),
            nn.Sigmoid()
        )

        # define loss function for regression
        self.loss_fn = nn.MSELoss()

    def forward(self, pixel_values, limbs, labels=None):
        # forward pass through the backbone
        outputs = self.image_model(pixel_values)
        hidden_states = outputs.last_hidden_state
        image_features = hidden_states[:, 0, :]

        # get the one hot encoded limb
        limbs_one_hot = torch.nn.functional.one_hot(limbs, num_classes=4).float()

        # pad one-hot with 4 zeros so the resulting vector is divisible by nheads = 2, 4, and 8
        padding = (0, 4)  # (padding_left, padding_right) for the last dimension
        limbs_one_hot_padded = torch.nn.functional.pad(limbs_one_hot, padding)

        # concatenate image features with one-hot limb
        combined_features = torch.cat((image_features, limbs_one_hot_padded), dim=-1)

        # forward pass through the regression head
        coords = self.regression_head(combined_features)
        return coords


def load_location_models(img_model_name, hands_model_name, feet_model_name, device):
    # load image model
    image_model = AutoModel.from_pretrained(img_model_name).to(device)

    # load hands model
    hands_config = AutoConfig.from_pretrained(hands_model_name)
    hands_processor = AutoProcessor.from_pretrained(hands_model_name)
    hands_model = LimbXYModel(
        image_model=image_model,
        config=hands_config,
        num_coordinates=2
    )
    file = hf_hub_download(repo_id=hands_model_name, filename="model.safetensors")
    state_dict = load_file(file)

    # update state dict to match custom model keys
    hands_model.load_state_dict(
        {key: value for key, value in state_dict.items()},
        strict=False,
    )
    hands_model.eval()

    # load foot model
    feet_config = AutoConfig.from_pretrained(feet_model_name)
    feet_processor = AutoProcessor.from_pretrained(feet_model_name)
    feet_model = LimbXYModel(
        image_model=image_model,
        config=feet_config,
        num_coordinates=2
    )
    file = hf_hub_download(repo_id=feet_model_name, filename="model.safetensors")
    state_dict = load_file(file)

    # update state dict to match custom model keys
    feet_model.load_state_dict(
        {key: value for key, value in state_dict.items()},
        strict=False,
    )

    feet_model.eval()

    return (hands_model, hands_processor), (feet_model, feet_processor)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_model_name = 'c14kevincardenas/ClimBEiT_single_frame'
    hands_model_name = 'c14kevincardenas/limbxy_hands'
    feet_model_name = 'c14kevincardenas/limbxy_feet'
    print('Loading Limb Prediction Models...')
    (hands_model, hands_processor), (feet_model, feet_processor) = load_location_models(img_model_name,
                                                                                        hands_model_name,
                                                                                        feet_model_name,
                                                                                        device)

    # Verification print statements
    print("\n=== Model Loading Verification ===")
    print(f"\nLimbXY Hands Model Loaded: {hands_model_name}")
    print(f"Processor Config: {hands_processor}")
    print(f"\nLimbXY Feet Model Loaded: {feet_model_name}")
    print(f"Processor Config: {feet_processor}")

    # Check model structure
    print("\n=== Model Structures ===")
    print(f"LimbXY Hands Model:\n{hands_model}")
    print(f"Image Model:\n{hands_model.image_model}")
    print(f"LimbXY Feet Model:\n{feet_model}")
    print(f"Image Model:\n{feet_model.image_model}")

    # Check device assignment
    print("\n=== Device Assignment ===")
    print(f"LimbXY Hands Model Device: {next(hands_model.parameters()).device}")
    print(f"LimbXY Feet Model Device: {next(feet_model.parameters()).device}")

    print("\nModels loaded successfully!")
