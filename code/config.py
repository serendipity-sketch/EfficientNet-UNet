import time
import torch
import torch.nn as nn
import pandas as pd
import os
import torchvision.transforms.functional as ff
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from torchvision import models
import six
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
# from model.manet import MANet
# from model.segnet import segnet
from model.linkent import linknet
from model.efficientunet import EfficientUNet
from datetime import datetime
import matplotlib.pyplot as plt
import tifffile as tiff
import random
import cv2
import albumentations as album
import warnings
from utils.LabelProcessor import LabelProcessor
warnings.filterwarnings("ignore")

plt.rcParams['font.family']='SimHei'

#设置全部随机种子
seed = 701
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
TRAIN_ROOT = r"/workspace/data/code/datasets/train/images"
TRAIN_LABEL = r"/workspace/data/code/datasets/train/labels"
VAL_ROOT = r"/workspace/data/code/datasets/val/images"
VAL_LABEL = r"/workspace/data/code/datasets/val/labels"
TEST_ROOT = "/workspace/data/code/datasets/test/images"
TEST_LABEL = "/workspace/data/code/datasets/test/labels"

# 类别字典文件路径
class_dict_path = "./datasets/class_dict.csv"
if not os.path.exists(class_dict_path):
    class_dict_data = {
        'name': ['background', 'foreground'],
        'r': [0, 255],
        'g': [0, 255],
        'b': [0, 255]
    }
    df = pd.DataFrame(class_dict_data)
    os.makedirs(os.path.dirname(class_dict_path), exist_ok=True)
    df.to_csv(class_dict_path, index=False)
    print(f"已创建默认类别字典文件: {class_dict_path}")

BATCH_SIZE = 4
crop_size = 256
val_size=256
class_num=2
LR=0.0001
EPOCH_NUMBER=100
WEIGHT_DECAY=0.01
LR_SCHEDULER_FACTOR=0.5
LR_SCHEDULER_PATIENCE=10
EARLY_STOPPING_PATIENCE=20
EARLY_STOPPING_MIN_DELTA=0.001

usemodel = EfficientUNet(
        encoder_name='efficientnet-b4',
        num_classes=class_num,
        use_mixed_activation=True
    )
LOG_SAVE_DIR = "./output"
dir="./pre-output"
best_pth="./output/best_ph.pth"
last_pth="./last_ph.pth"
