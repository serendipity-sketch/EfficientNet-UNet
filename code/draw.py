import os
import pandas as pd
import matplotlib.pyplot as plt
from config import *

def plot_learning_curves():
    fontweight = 16

    temp_net = usemodel
    net_name = temp_net.__class__.__name__
    del temp_net

    print(f"[{net_name}] 开始绘制曲线图...")
    log_dir = LOG_SAVE_DIR
    train_log_path = os.path.join(log_dir, f"{net_name}_train_log.csv")
    val_log_path = os.path.join(log_dir, f"{net_name}_val_log.csv")

    if not os.path.exists(train_log_path) or not os.path.exists(val_log_path):
        print(f"错误：找不到日志\n{train_log_path}\n{val_log_path}")
        return False

    df_train = pd.read_csv(train_log_path)
    df_val = pd.read_csv(val_log_path)

    plt.figure(figsize=(15, 6))
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(df_train['epoch'], df_train['loss'], label='训练集 Loss', color='#3498DB', linewidth=2)
    plt.plot(df_val['epoch'], df_val['loss'], label='验证集 Loss', color='#E74C3C', linestyle='--', linewidth=2)
    plt.title(f'{net_name} Loss 曲线', fontsize=fontweight)
    plt.xticks(fontsize=fontweight)
    plt.yticks(fontsize=fontweight)
    plt.xlabel('训练轮数 (Epoch)', fontsize=fontweight)
    plt.ylabel('损失值', fontsize=fontweight)
    plt.legend(fontsize=fontweight - 2)
    plt.grid(True, alpha=0.3)

    # mIoU 曲线
    plt.subplot(1, 2, 2)
    plt.plot(df_train['epoch'], df_train['miou'], label='训练集 mIoU', color='#2ECC71', linewidth=2)
    plt.plot(df_val['epoch'], df_val['miou'], label='验证集 mIoU', color='#9B59B6', linestyle='--', linewidth=2)
    plt.title(f'{net_name} mIoU 曲线', fontsize=fontweight)
    plt.xticks(fontsize=fontweight)
    plt.yticks(fontsize=fontweight)
    plt.xlabel('训练轮数 (Epoch)', fontsize=fontweight)
    plt.ylabel('mIoU', fontsize=fontweight)
    plt.legend(fontsize=fontweight - 2)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(log_dir, f"{net_name}_metrics_summary.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"绘图完成！已保存到：{save_path}")
    return True

if __name__ == '__main__':
    plot_learning_curves()