import os
import glob
import time
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from utils.augmentation import get_validation_augmentation
from utils.Dataset import MyDataset
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_dir = config.dir
os.makedirs(output_dir, exist_ok=True)

test_img_paths = sorted(glob.glob(os.path.join(config.TEST_ROOT, "*.tif*")))
test_dataset = MyDataset(
    [config.TEST_ROOT, config.TEST_LABEL],
    transform=get_validation_augmentation(config.val_size)
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

print(f"共加载测试图像: {len(test_img_paths)} 张")

def load_checkpoint(model, ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))

    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model

model = config.usemodel.to(device)
model.eval()
load_checkpoint(model, config.best_pth)
print("模型加载完成，开始预测...")

total_cost = 0.0

with torch.no_grad():
    for idx, sample in enumerate(test_loader):
        start_time = time.time()
        img = sample["img"].to(device)
        img_name = os.path.basename(test_img_paths[idx])
        base_name = os.path.splitext(img_name)[0]
        out = model(img)
        pred = (torch.sigmoid(out) > 0.5).squeeze().cpu().numpy().astype(np.uint8)


        save_mask = Image.fromarray(pred)
        save_mask.save(os.path.join(output_dir, f"{base_name}_mask.png"))
        save_mask.save(os.path.join(output_dir, f"{base_name}_mask.tif"))

        cost = time.time() - start_time
        total_cost += cost
        print(f"[{idx+1}/{len(test_loader)}] {base_name} | 耗时: {cost:.3f}s")

print(f"\n总耗时: {total_cost:.2f}s")
print(f"预测结果保存至: {output_dir}")