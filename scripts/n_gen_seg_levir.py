import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm

def generate_change_mask(dir_A, dir_B, dst_dir, change_threshold=1, edge_margin=3, use_morph=False):
    """
    生成像素级别的变化掩码，自动排除边缘伪变化。

    参数说明：
    - dir_A: 时间点 A_2 图像文件夹路径
    - dir_B: 时间点 B_2 图像文件夹路径
    - dst_dir: 输出变化掩码保存路径
    - change_threshold: 像素差阈值，默认为1
    - edge_margin: 去除边缘伪差异的边缘宽度（单位：像素）
    - use_morph: 是否启用形态学开运算去噪（默认False）
    """
    os.makedirs(dst_dir, exist_ok=True)

    for file_name in tqdm(os.listdir(dir_A), desc="Generating Change Masks"):
        if not file_name.lower().endswith('.png'):
            continue

        path_A = osp.join(dir_A, file_name)
        path_B = osp.join(dir_B, file_name)

        img_A = cv2.imread(path_A, cv2.IMREAD_GRAYSCALE)
        img_B = cv2.imread(path_B, cv2.IMREAD_GRAYSCALE)

        if img_A is None or img_B is None:
            print(f"❌ 图像读取失败: {file_name}")
            continue

        if img_A.shape != img_B.shape:
            print(f"⚠️ 图像尺寸不一致: {file_name}")
            continue

        # 计算像素差异图
        diff = np.abs(img_A.astype(np.int16) - img_B.astype(np.int16))

        # 生成变化掩码（二值图）
        change_mask = (diff >= change_threshold).astype(np.uint8) * 255

        # 去除边缘伪差异
        h, w = change_mask.shape
        change_mask[:edge_margin, :] = 0
        change_mask[-edge_margin:, :] = 0
        change_mask[:, :edge_margin] = 0
        change_mask[:, -edge_margin:] = 0

        # 可选：形态学开运算去小噪声
        if use_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            change_mask = cv2.morphologyEx(change_mask, cv2.MORPH_OPEN, kernel)

        # 保存变化掩码图像
        save_path = osp.join(dst_dir, file_name)
        cv2.imwrite(save_path, change_mask)

    print("✅ 所有变化掩码生成完毕。")

if __name__ == "__main__":
    # 替换为你的路径
    dir_A = '/APE_output/ape_selftrain_whu_cd_pseudo_label/A'
    dir_B = '/APE_output/ape_selftrain_whu_cd_pseudo_label/B'
    dst_dir = '/gen_cd_label/whu_self_train_pixel_level_change_mask'

    # 生成掩码
    generate_change_mask(
        dir_A=dir_A,
        dir_B=dir_B,
        dst_dir=dst_dir,
        change_threshold=1,     # 有像素差异就视为变化
        edge_margin=3,          # 去除图像边缘3像素的变化
        use_morph=True          # 启用形态学处理去小噪点
    )
