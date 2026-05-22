import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import io
import csv

os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
warnings.filterwarnings("ignore", message="A new version of Albumentations is available.*")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from config import *
from utils.DiceFocalLoss import DiceFocalLoss
from model.efficientunet import EfficientUNet
from utils.Dataset import MyDataset
from utils.augmentation import get_training_augmentation, get_validation_augmentation
from utils.Metrics import eval_semantic_segmentation



class EarlyStopping():
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_miou):
        score = val_miou
        if self.best_score is None:
            self.best_score = score
        elif score <= self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


def log_single_epoch(epoch, miou, pixel_accuracy, loss, model_name, filename):
    log_path = os.path.join(LOG_SAVE_DIR, filename) if 'LOG_SAVE_DIR' in globals() else filename
    file_exists = os.path.isfile(log_path)

    with open(log_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['epoch', 'miou', 'pixel_acc', 'loss', 'model_name'])
        writer.writerow([epoch, round(miou, 5), round(pixel_accuracy, 5), round(loss, 5), model_name])


if __name__ == "__main__":

    model_name = "EfficientUNet"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    train_dataset = MyDataset(
        [TRAIN_ROOT, TRAIN_LABEL],
        get_training_augmentation(crop_size),

    )
    val_dataset = MyDataset(
        [VAL_ROOT, VAL_LABEL],
        get_validation_augmentation(val_size),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    print(f"训练集:{len(train_dataset)} 验证集:{len(val_dataset)}")

    model = EfficientUNet(
        encoder_name='efficientnet-b4',
        num_classes=class_num,
        use_mixed_activation=True
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"使用 {torch.cuda.device_count()} 个GPU")

    criterion = DiceFocalLoss(weight_dice=0.5, weight_focal=0.5, class_num=class_num).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=1e-7
    )

    # 训练配置
    best_miou = 0
    train_loss_list, val_loss_list = [], []
    train_miou_list, val_miou_list = [], []
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA
    )

    start_time = time.time()
    for epoch in range(1, EPOCH_NUMBER + 1):
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch}/{EPOCH_NUMBER}] | LR: {current_lr:.5f}'),
        # -------------------------- 训练 --------------------------
        model.train()
        total_train_loss = 0.0
        train_pred_list, train_true_list = [], []

        for i, sample in enumerate(train_loader):
            img = sample["img"].to(device)
            label = sample["label"].to(device)

            optimizer.zero_grad()
            out = model(img)

            if out.shape[-2:] != label.shape[-2:]:
                out = F.interpolate(out, size=label.shape[-2:], mode='bilinear', align_corners=True)

            loss = criterion(out, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()

            # 预测
            with torch.no_grad():
                if class_num == 1:
                    pred = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(np.uint8).squeeze(1)
                else:
                    pred = out.argmax(dim=1).cpu().numpy().astype(np.uint8)
                true = label.cpu().numpy().astype(np.uint8)

            train_pred_list.extend(pred)
            train_true_list.extend(true)

            # 日志
            if (i + 1) % max(1, len(train_loader) // 10) == 0:
                print(f'| Batch [{i + 1}/{len(train_loader)}] | Loss: {loss.item():.6f} |')


        avg_train_loss = total_train_loss / len(train_loader)
        train_metrics = eval_semantic_segmentation(
            train_pred_list, train_true_list,
            n_class=class_num, ignore_label=255
        )
        train_miou = train_metrics['miou']
        train_pixel_acc = train_metrics['pixel_accuracy']

        log_single_epoch(
            epoch=epoch,
            miou=train_miou,
            pixel_accuracy=train_pixel_acc,
            loss=avg_train_loss,
            model_name=model_name,
            filename=f"{model_name}_train_log.csv"
        )

        # -------------------------- 验证 --------------------------
        model.eval()
        total_val_loss = 0.0
        val_pred_list, val_true_list = [], []

        with torch.no_grad():
            for sample in val_loader:
                img = sample['img'].to(device)
                label = sample['label'].to(device)
                out = model(img)

                if out.shape[-2:] != label.shape[-2:]:
                    out = F.interpolate(out, size=label.shape[-2:], mode='bilinear', align_corners=True)

                loss = criterion(out, label)
                total_val_loss += loss.item()

                if class_num == 1:
                    pred = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(np.uint8).squeeze(1)
                else:
                    pred = out.argmax(dim=1).cpu().numpy().astype(np.uint8)
                true = label.cpu().numpy().astype(np.uint8)

                val_pred_list.extend(pred)
                val_true_list.extend(true)

        avg_val_loss = total_val_loss / len(val_loader)
        val_metrics = eval_semantic_segmentation(
            val_pred_list, val_true_list,
            n_class=class_num, ignore_label=255
        )
        val_miou = val_metrics['miou']
        val_pixel_acc = val_metrics['pixel_accuracy']

        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)
        train_miou_list.append(train_miou)
        val_miou_list.append(val_miou)

        # 学习率
        scheduler.step(val_miou)

        log_single_epoch(
            epoch=epoch,
            miou=val_miou,
            pixel_accuracy=val_pixel_acc,
            loss=avg_val_loss,
            model_name=model_name,
            filename=f"{model_name}_val_log.csv"
        )

        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_miou': best_miou,
            }, best_pth)

            print(f"保存最佳模型 | mIoU: {val_miou:.5f}")
        print(f"| Train Loss: {avg_train_loss:.5f} | Train mIoU: {train_miou:.5f} |")
        print(f"| Val Loss: {avg_val_loss:.5f} | Val mIoU: {val_miou:.5f} |")
        if 'pixel_accuracy' in val_metrics:
            print(f"| Pixel Acc: {val_metrics['pixel_accuracy']:.5f} |")

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, last_pth)



        # 早停
        if early_stopping(val_miou):
            print(f"早停触发，Epoch: {epoch}")
            break


    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    print("训练完成！")
    print(f"总耗时: {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    print(f"最佳 mIoU: {best_miou:.5f}")