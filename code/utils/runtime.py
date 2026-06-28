from pathlib import Path

import numpy as np
import torch

import config


def build_model(pretrained_encoder=False):
    name = config.MODEL_NAME.lower()

    if name == "efficientunet":
        from model.efficientunet import EfficientUNet
        return EfficientUNet(
            encoder_name=config.ENCODER_NAME,
            num_classes=config.NUM_CLASSES,
            use_mixed_activation=config.USE_MIXED_ACTIVATION,
            pretrained_encoder=pretrained_encoder,
        )
    if name == "linknet":
        from model.linkent import LinkNet
        return LinkNet(n_classes=config.NUM_CLASSES, pretrained=pretrained_encoder)
    if name == "manet":
        from model.manet import MANet
        return MANet(num_classes=config.NUM_CLASSES, pretrained=pretrained_encoder)
    if name == "segnet":
        from model.segnet import SegNet
        return SegNet(num_classes=config.NUM_CLASSES)
    if name == "unetplusplus":
        from model.unetplusplus import UNetPlusPlus
        return UNetPlusPlus(num_classes=config.NUM_CLASSES)

    raise ValueError(
        f"Unsupported model: {config.MODEL_NAME}. "
        "Choose EfficientUNet, LinkNet, MANet, SegNet or UNetPlusPlus."
    )


def load_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)
    return checkpoint


def checkpoint_state(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    return model.state_dict()


def logits_to_predictions(logits):
    if config.NUM_CLASSES == 1:
        return (torch.sigmoid(logits[:, 0]) >= 0.5).to(torch.uint8)
    return logits.argmax(dim=1).to(torch.uint8)


def colorize_mask(mask):
    mask = np.asarray(mask, dtype=np.uint8)
    if config.NUM_CLASSES == 2:
        return mask * 255
    scale = 255 / max(config.NUM_CLASSES - 1, 1)
    return np.rint(mask * scale).astype(np.uint8)
