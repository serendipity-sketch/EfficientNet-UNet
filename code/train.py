import argparse
import csv
import os
import random
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import config
from utils.Dataset import SegmentationDataset
from utils.DiceFocalLoss import DiceFocalLoss
from utils.Metrics import calc_semantic_segmentation_confusion, metrics_from_confusion
from utils.augmentation import get_training_augmentation, get_validation_augmentation
from utils.runtime import build_model, checkpoint_state, logits_to_predictions


class EarlyStopping:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def __call__(self, score):
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument(
        "--no-pretrained", action="store_true",
    )
    return parser.parse_args()


def append_log(filename, epoch, metrics, loss):
    path = config.OUTPUT_DIR / filename
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not exists:
            writer.writerow(["epoch", "miou", "pixel_acc", "loss", "model_name"])
        writer.writerow([
            epoch, round(metrics["miou"], 5), round(metrics["pixel_accuracy"], 5),
            round(loss, 5), config.MODEL_NAME,
        ])


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    confusion = np.zeros((config.NUM_CLASSES, config.NUM_CLASSES), dtype=np.int64)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for step, sample in enumerate(loader, start=1):
            images = sample["img"].to(device)
            labels = sample["label"].to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, labels.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            predictions = logits_to_predictions(logits).detach().cpu().numpy()
            truths = labels.detach().cpu().numpy()
            confusion += calc_semantic_segmentation_confusion(
                predictions, truths, config.NUM_CLASSES
            )

            if training and step % max(1, len(loader) // 10) == 0:
                print(f"  batch {step}/{len(loader)} | loss {loss.item():.6f}")

    return total_loss / len(loader), metrics_from_confusion(confusion)


def save_checkpoint(path, epoch, model, optimizer, val_miou):
    torch.save({
        "epoch": epoch,
        "model_state_dict": checkpoint_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_miou": val_miou,
    }, path)


def main():
    args = parse_args()
    seed_everything(config.SEED)
    config.MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_set = SegmentationDataset(
        config.TRAIN_IMAGE_DIR, config.TRAIN_LABEL_DIR,
        get_training_augmentation(config.IMAGE_SIZE), config.NUM_CLASSES,
    )
    val_set = SegmentationDataset(
        config.VAL_IMAGE_DIR, config.VAL_LABEL_DIR,
        get_validation_augmentation(config.IMAGE_SIZE), config.NUM_CLASSES,
    )
    train_loader = DataLoader(train_set, config.BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_set, config.BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练集: {len(train_set)}，验证集: {len(val_set)}")

    use_pretrained = config.PRETRAINED_ENCODER and not args.no_pretrained
    model = build_model(pretrained_encoder=use_pretrained).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"使用 {torch.cuda.device_count()} 块 GPU")

    criterion = DiceFocalLoss(class_num=config.NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=config.LR_SCHEDULER_FACTOR,
        patience=config.LR_SCHEDULER_PATIENCE, min_lr=1e-7,
    )
    early_stopping = EarlyStopping(
        config.EARLY_STOPPING_PATIENCE, config.EARLY_STOPPING_MIN_DELTA
    )
    best_miou = -np.inf
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch}/{args.epochs}] | LR: {optimizer.param_groups[0]['lr']:.6g}")
        train_loss, train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device)
        scheduler.step(val_metrics["miou"])

        append_log(f"{config.MODEL_NAME}_train_log.csv", epoch, train_metrics, train_loss)
        append_log(f"{config.MODEL_NAME}_val_log.csv", epoch, val_metrics, val_loss)
        save_checkpoint(config.LAST_CHECKPOINT, epoch, model, optimizer, val_metrics["miou"])
        if val_metrics["miou"] > best_miou:
            best_miou = val_metrics["miou"]
            save_checkpoint(config.BEST_CHECKPOINT, epoch, model, optimizer, best_miou)
            print(f"  已保存最佳模型 | mIoU: {best_miou:.5f}")

        print(f"  train loss {train_loss:.5f} | mIoU {train_metrics['miou']:.5f}")
        print(f"  val   loss {val_loss:.5f} | mIoU {val_metrics['miou']:.5f}")
        if early_stopping(val_metrics["miou"]):
            print(f"早停触发，Epoch: {epoch}")
            break

    elapsed = time.time() - started
    print(f"训练完成，用时 {elapsed / 3600:.2f} 小时，最佳 mIoU: {best_miou:.5f}")


if __name__ == "__main__":
    main()
