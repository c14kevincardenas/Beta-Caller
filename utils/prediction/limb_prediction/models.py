import torch
import numpy as np
from torch import nn
from transformers import (
    DetrForObjectDetection,
    DetrImageProcessor,
    AutoModelForImageClassification,
    AutoProcessor,
    AutoModel,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: (batch_size, seq_len, d_model)
        pe shape: (seq_len, d_model) -> unsqueeze to (1, seq_len, d_model)
        """
        seq_len = x.size(1)
        pe = self.pe[:seq_len, :].unsqueeze(0)  # Shape: (1, seq_len, d_model)

        return x + pe  # Broadcast across batch dimension


class ImageSequenceModel(nn.Module):
    def __init__(self, image_model, num_classes, d_model, onehot_dim, nhead=4, num_layers=1):
        super(ImageSequenceModel, self).__init__()
        self.image_model = image_model

        # freeze the pre-trained layers
        for param in self.image_model.parameters():
            param.requires_grad = False

        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_encoder_limb = PositionalEncoding(4)
        encoder_layers = nn.TransformerEncoderLayer(d_model+onehot_dim, nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model+onehot_dim, num_classes)

    def forward(self, pixel_values, prev_labels):
        # add a batch dimension of 1 at the front
        pixel_values = pixel_values.unsqueeze(0)
        prev_labels = prev_labels.unsqueeze(0)

        # pixel_values shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = pixel_values.size()

        # flatten the batch and sequence dimensions to process all images at once
        pixel_values = pixel_values.view(batch_size * seq_len, c, h, w)

        # extract image features using the pre-trained model
        outputs = self.image_model(pixel_values=pixel_values)
        image_features = outputs.pooler_output  # extracting from the penultimate layer

        # reshape back to (batch_size, seq_len, d_model)
        image_features = image_features.view(batch_size, seq_len, -1)

        # apply positional encoding
        image_features = self.pos_encoder(image_features)

        # one hot encode prev labels
        prev_labels_one_hot = torch.nn.functional.one_hot(prev_labels, num_classes=4).float()

        # apply positional encoding to limbs
        prev_labels_one_hot = self.pos_encoder_limb(prev_labels_one_hot)

        # pad one-hot with 4 zeros so the resulting vector is divisible by nheads = 2, 4, and 8
        padding = (0, 4)
        prev_labels_one_hot_padded = torch.nn.functional.pad(prev_labels_one_hot, padding)

        # concatenate image features with one-hot limb
        combined_features = torch.cat((image_features, prev_labels_one_hot_padded), dim=-1)

        # forward pass through transformer encoder
        sequence_output = self.transformer_encoder(combined_features)

        # forward pass through the classification head
        logits = self.fc(sequence_output.mean(dim=1))

        return logits


class StudentModel(nn.Module):
    def __init__(self, student_model_name, dff=512, nhead=4, num_layers=1, num_classes=4, onehot_dim=8):
        super(StudentModel, self).__init__()
        self.image_model = AutoModel.from_pretrained(student_model_name)

        # find hidden state dimension
        try:
            if hasattr(self.image_model.config, "hidden_size"):
                d_model = self.image_model.config.hidden_size  # DeiT, BEiT
            elif hasattr(self.image_model.config, "neck_hidden_sizes"):
                d_model = self.image_model.config.neck_hidden_sizes[-1]  # MobileViT
            elif hasattr(self.image_model.config, "embed_dim"):
                d_model = self.image_model.config.embed_dim  # Swin
            else:
                raise ValueError("Unable to determine `d_model` from the image model's configuration.")
        except AttributeError as e:
            raise ValueError(f"Unexpected configuration structure for the image model: {e}")
        # print(f'\tStudent Image Model Dimension: {d_model}')

        self.pos_encoder = PositionalEncoding(d_model)
        self.pos_encoder_limb = PositionalEncoding(num_classes)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model + onehot_dim,
            nhead=nhead,
            dim_feedforward=dff,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model+onehot_dim, num_classes)

    def forward(self, images, prev_labels, labels=None):
        pixel_values = images
        # add a batch dimension of 1 at the front
        pixel_values = pixel_values.unsqueeze(0)
        prev_labels = prev_labels.unsqueeze(0)

        # pixel_values shape: (batch_size, seq_len, c, h, w)
        batch_size, seq_len, c, h, w = pixel_values.size()
        pixel_values = pixel_values.view(batch_size * seq_len, c, h, w)
        outputs = self.image_model(pixel_values=pixel_values)
        image_features = outputs.pooler_output
        image_features = image_features.view(batch_size, seq_len, -1)

        # positionally encode image features
        image_features = self.pos_encoder(image_features)

        # get the one hot encoded limb
        prev_labels_one_hot = torch.nn.functional.one_hot(prev_labels, num_classes=4).float()

        # apply positional encoding to limbs
        prev_labels_one_hot = self.pos_encoder_limb(prev_labels_one_hot)

        # pad one-hot with 4 zeros so the resulting vector is divisible by nheads = 2, 4, and 8
        padding = (0, 4)
        prev_labels_one_hot_padded = torch.nn.functional.pad(prev_labels_one_hot, padding)

        # concatenate image features with one-hot limb
        combined_features = torch.cat((image_features, prev_labels_one_hot_padded), dim=-1)

        # forward pass through transformer encoder
        sequence_output = self.transformer_encoder(combined_features)

        # aggregate the sequence output and pass through the final classification layer
        logits = self.fc(sequence_output.mean(dim=1))

        return logits


def load_limb_models(detr_model_name, beit_model_name, climbeit_model_name, device):
    # load DETR model
    detr_model = DetrForObjectDetection.from_pretrained(detr_model_name).to(device)
    detr_processor = DetrImageProcessor.from_pretrained(detr_model_name)

    # load BEiT model
    beit_model = AutoModelForImageClassification.from_pretrained(beit_model_name).to(device)
    beit_processor = AutoProcessor.from_pretrained(beit_model_name)

    # load ClimBEiT model
    image_model = AutoModel.from_pretrained(beit_model_name).to(device)
    climbeit_processor = AutoProcessor.from_pretrained(climbeit_model_name)
    climbeit_model = ImageSequenceModel(
        image_model=image_model,
        num_classes=4,
        d_model=1024,
        onehot_dim=8,
        nhead=4,
        num_layers=1,
    )
    file = hf_hub_download(repo_id=climbeit_model_name, filename="model.safetensors")
    state_dict = load_file(file)

    # import state dict to model
    missing_keys, unexpected_keys = climbeit_model.load_state_dict(state_dict, strict=False)

    # print(f"Missing keys: {missing_keys}")
    # print(f"Unexpected keys: {unexpected_keys}")

    climbeit_model.eval()

    return (detr_model, detr_processor), (beit_model, beit_processor), (climbeit_model, climbeit_processor)


def load_limb_models_student(detr_model_name, beit_model_name, image_model_name, model_name, device):
    # load DETR model
    detr_model = DetrForObjectDetection.from_pretrained(detr_model_name).to(device)
    detr_processor = DetrImageProcessor.from_pretrained(detr_model_name)

    # load BEiT model
    beit_model = AutoModelForImageClassification.from_pretrained(beit_model_name).to(device)
    beit_processor = AutoProcessor.from_pretrained(beit_model_name)

    # load student model
    processor = AutoProcessor.from_pretrained(model_name)
    model = StudentModel(
        student_model_name=image_model_name,
        dff=512,
        nhead=4,
        num_layers=1,
    )
    file = hf_hub_download(repo_id=model_name, filename="model.safetensors")
    state_dict = load_file(file)

    # import state dict to model
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    # print(f"Missing keys: {missing_keys}")
    # print(f"Unexpected keys: {unexpected_keys}")

    model.eval()

    return (detr_model, detr_processor), (beit_model, beit_processor), (model, processor)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detr_model_name = "facebook/detr-resnet-50"
    beit_model_name = "c14kevincardenas/ClimBEiT_single_frame"
    climbeit_model_name = "c14kevincardenas/ClimBEiTv2"
    print('Loading Limb Prediction Models...')
    (detr_model, detr_processor), (beit_model, beit_processor), (climbeit_model, climbeit_processor) = load_limb_models(
        detr_model_name, beit_model_name, climbeit_model_name, device)

    # Verification print statements
    print("\n=== Model Loading Verification ===")
    print(f"DETR Model Loaded: {detr_model_name}")
    print(f"DETR Processor Attributes: {detr_processor}")

    print(f"\nBEiT Model Loaded: {beit_model_name}")
    print(f"BEiT Processor Config: {beit_processor}")

    print(f"\nClimBEiT Model Loaded: {climbeit_model_name}")
    print(f"ClimBEiT Model Parameters (requires_grad=False):")
    for name, param in climbeit_model.named_parameters():
        if not param.requires_grad:
            print(f"- {name}")

    # Check model structure
    print("\n=== Model Structures ===")
    print(f"ClimBEiT Model:\n{climbeit_model}")
    print(f"Image Model:\n{climbeit_model.image_model}")

    # Check device assignment
    print("\n=== Device Assignment ===")
    print(f"DETR Model Device: {next(detr_model.parameters()).device}")
    print(f"BEiT Model Device: {next(beit_model.parameters()).device}")
    print(f"ClimBEiT Model Device: {next(climbeit_model.parameters()).device}")

    print("\nModels loaded successfully!")
