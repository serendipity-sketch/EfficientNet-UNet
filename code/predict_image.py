import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

import config
from utils.Dataset import PredictionDataset
from utils.augmentation import get_validation_augmentation
from utils.runtime import build_model, colorize_mask, load_checkpoint, logits_to_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=config.TEST_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=config.PREDICTION_DIR)
    parser.add_argument("--checkpoint", type=Path, default=config.BEST_CHECKPOINT)
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PredictionDataset(args.input, get_validation_augmentation(config.IMAGE_SIZE))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = build_model().to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()
    print(f"已加载 {len(dataset)} 张影像，使用设备: {device}")

    started = time.time()
    with torch.no_grad():
        for index, sample in enumerate(loader, start=1):
            logits = model(sample["img"].to(device))
            prediction = logits_to_predictions(logits)[0].cpu().numpy()
            width = int(sample["original_size"][0].item())
            height = int(sample["original_size"][1].item())
            if prediction.shape != (height, width):
                prediction = F.interpolate(
                    torch.from_numpy(prediction)[None, None].float(),
                    size=(height, width), mode="nearest",
                )[0, 0].to(torch.uint8).numpy()

            stem = Path(sample["name"][0]).stem
            Image.fromarray(colorize_mask(prediction)).save(args.output / f"{stem}_mask.png")
            print(f"[{index}/{len(dataset)}] {stem}")

    print(f"预测完成，用时 {time.time() - started:.2f} 秒；结果目录: {args.output.resolve()}")


if __name__ == "__main__":
    main()
