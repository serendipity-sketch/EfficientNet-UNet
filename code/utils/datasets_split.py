import os
import shutil
import random
def split_dataset_corrected():
    base_dir = r"D:\海伦\数据"
    augmented_dir = os.path.join(base_dir, "augmented_data")
    output_base_dir = os.path.join(base_dir, "datasets")

    # 创建输出目录
    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(os.path.join(output_base_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(output_base_dir, split, 'labels'), exist_ok=True)

    # 获取所有图像文件
    image_files = [f for f in os.listdir(os.path.join(augmented_dir, 'images'))
                   if f.endswith('.tif')]

    print(f"增强后总文件数: {len(image_files)}")

    # 直接按文件划分，而不是按基础名称
    random.shuffle(image_files)

    # 计算划分数量
    total_files = len(image_files)
    train_count = int(total_files * 0.7)  # 70%
    val_count = int(total_files * 0.2)  # 20%
    test_count = total_files - train_count - val_count  # 10%

    # 划分文件
    train_files = image_files[:train_count]
    val_files = image_files[train_count:train_count + val_count]
    test_files = image_files[train_count + val_count:]

    print(f"总文件数: {total_files}")
    print(f"训练集文件数: {len(train_files)} ({(len(train_files) / total_files) * 100:.1f}%)")
    print(f"验证集文件数: {len(val_files)} ({(len(val_files) / total_files) * 100:.1f}%)")
    print(f"测试集文件数: {len(test_files)} ({(len(test_files) / total_files) * 100:.1f}%)")

    # 复制文件到相应目录
    def copy_file_list(file_list, split_name):
        copied_count = 0
        for img_file in file_list:
            # 复制图像
            src_img = os.path.join(augmented_dir, 'images', img_file)
            dst_img = os.path.join(output_base_dir, split_name, 'images', img_file)
            shutil.copy2(src_img, dst_img)

            # 复制标签
            label_file = img_file
            src_label = os.path.join(augmented_dir, 'labels', label_file)
            dst_label = os.path.join(output_base_dir, split_name, 'labels', label_file)

            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
                copied_count += 1

        print(f"{split_name}集: 复制了 {copied_count} 个文件")
        return copied_count

    train_count = copy_file_list(train_files, 'train')
    val_count = copy_file_list(val_files, 'val')
    test_count = copy_file_list(test_files, 'test')

    total_copied = train_count + val_count + test_count
    print(f"\n数据集划分完成！")
    print(f"最终分布: 训练集 {train_count}, 验证集 {val_count}, 测试集 {test_count}")


if __name__ == "__main__":
    split_dataset_corrected()
