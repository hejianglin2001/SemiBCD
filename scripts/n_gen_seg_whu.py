import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm


def generate_change_mask(dir_A, dir_B, dst_dir):
    """
    生成像素级别的变化掩码。

    :param dir_A: 时间点A的图像目录路径
    :param dir_B: 时间点B的图像目录路径
    :param dst_dir: 输出变化掩码的目标目录路径
    """
    os.makedirs(dst_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(dir_A)):
        if file_name.lower().endswith('.png'):
            # 读取A和B的时间点图像
            img_A = cv2.imread(osp.join(dir_A, file_name), cv2.IMREAD_GRAYSCALE)
            img_B = cv2.imread(osp.join(dir_B, file_name), cv2.IMREAD_GRAYSCALE)

            if img_A is None or img_B is None:
                print(f"Failed to load one of the images: {file_name}")
                continue

            # 计算绝对差异
            diff = np.abs(img_A.astype(np.int16) - img_B.astype(np.int16))

            # 定义变化阈值（可以根据需要调整）
            change_threshold = 1  # 如果有任何差异，则认为是变化

            # 生成变化掩码
            change_mask = (diff >= change_threshold).astype(np.uint8) * 255

            # 保存变化掩码图像
            cv2.imwrite(osp.join(dst_dir, file_name), change_mask)


if __name__ == "__main__":
    dir_A = '/APE_output/levir_cd_pseudo_label_unce/A_255'
    dir_B = '/APE_output/levir_cd_pseudo_label_unce/B_255'
    dst_dir = '/gen_cd_label/levir_self_train_pixel_level_change_mask'

    generate_change_mask(dir_A, dir_B, dst_dir)

    print("Change detection masks generation complete.")
