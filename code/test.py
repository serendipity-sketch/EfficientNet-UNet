import argparse
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from utils.Dataset import SegmentationDataset
from utils.Metrics import calc_semantic_segmentation_confusion, metrics_from_confusion
from utils.augmentation import get_validation_augmentation
from utils.runtime import build_model, load_checkpoint, logits_to_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=str(config.BEST_CHECKPOINT))
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SegmentationDataset(
        config.TEST_IMAGE_DIR, config.TEST_LABEL_DIR,
        get_validation_augmentation(config.IMAGE_SIZE), config.NUM_CLASSES,
    )
    loader = DataLoader(dataset, config.BATCH_SIZE, shuffle=False, num_workers=0)
    model = build_model().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    confusion = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.int64)
    with torch.no_grad():
        for sample in loader:
            labels = sample["label"].to(device)
            logits = model(sample["img"].to(device))
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
            confusion += calc_semantic_segmentation_confusion(
                logits_to_predictions(logits).cpu().numpy(),
                labels.cpu().numpy(), config.NUM_CLASSES,
            )

    metrics = metrics_from_confusion(confusion)
    serializable = {
        key: value.tolist() if isinstance(value, np.ndarray) else float(value)
        for key, value in metrics.items()
    }
    config.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.TEST_RESULT.write_text(json.dumps(serializable, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"测试集 {len(dataset)} 张 | mIoU {metrics['miou']:.5f} | "
        f"PA {metrics['pixel_accuracy']:.5f} | F1 {metrics['f1']:.5f} | "
        f"Precision {metrics['precision']:.5f} | Recall {metrics['recall']:.5f}"
    )
    print(f"详细结果: {config.TEST_RESULT}")


if __name__ == "__main__":
    main()
