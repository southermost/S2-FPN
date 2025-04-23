import os
import random
import numpy as np
import cv2


def build_mosaics(mosaic_1: np.array, mosaic_2: np.array, type: str = "diag",
                  mask: np.array = np.zeros(())) -> np.array:
    if mosaic_1.shape != mosaic_2.shape:
        raise ValueError("Two images should have the same shape")
    h, w = mosaic_1.shape
    image = np.zeros_like(mosaic_1)
    if type == "diag":
        image[:h // 2, :w // 2] = mosaic_1[:h // 2, :w // 2]
        image[:h // 2, w // 2:] = mosaic_2[:h // 2, w // 2:]
        image[h // 2:, :w // 2] = mosaic_2[h // 2:, :w // 2]
        image[h // 2:, w // 2:] = mosaic_1[h // 2:, w // 2:]
    elif type == "upper_lower":
        print("Applying upper_lower mode.")
        # 上半部分来自 mosaic_1，下半部分来自 mosaic_2
        image[:h // 2, :] = mosaic_1[:h // 2, :]
        image[h // 2:, :] = mosaic_2[h // 2:, :]
    elif type == "left_right":
        image[:, :w // 2] = mosaic_1[:, :w // 2]
        image[:, w // 2:] = mosaic_2[:, w // 2:]
    elif type == "mask":
        image = np.where(mask == 0, mosaic_1, mosaic_2)
    return image


def add_salt_and_pepper_noise(image_shape, amount=0.2):
    """
    生成一个具有椒盐噪声的掩码。
    amount: 噪声的比例，越高则噪声越多。
    """
    mask = np.ones(image_shape, dtype=np.uint8)
    num_salt = int(np.ceil(amount * image_shape[0] * image_shape[1] * 0.5))
    num_pepper = int(np.ceil(amount * image_shape[0] * image_shape[1] * 0.5))

    # 添加盐噪声（白色噪声）
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image_shape]
    mask[salt_coords[0], salt_coords[1]] = 1

    # 添加胡椒噪声（黑色噪声）
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image_shape]
    mask[pepper_coords[0], pepper_coords[1]] = 0

    return mask


def process_images(image_folder, label_folder, output_image_folder, output_label_folder):
    os.makedirs(output_image_folder, exist_ok=True)
    os.makedirs(output_label_folder, exist_ok=True)

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    label_files = [f.replace('.jpg', '.png') for f in image_files]

    for i, image_file in enumerate(image_files):
        image_path = os.path.join(image_folder, image_file)
        label_path = os.path.join(label_folder, label_files[i])

        mosaic_1 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label_1 = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        other_image_file = random.choice([f for f in image_files if f != image_file])
        other_image_path = os.path.join(image_folder, other_image_file)
        other_label_path = os.path.join(label_folder, other_image_file.replace('.jpg', '.png'))

        mosaic_2 = cv2.imread(other_image_path, cv2.IMREAD_GRAYSCALE)
        label_2 = cv2.imread(other_label_path, cv2.IMREAD_GRAYSCALE)

        if mosaic_1.shape != mosaic_2.shape:
            mosaic_2 = cv2.resize(mosaic_2, (mosaic_1.shape[1], mosaic_1.shape[0]))
            label_2 = cv2.resize(label_2, (label_1.shape[1], label_1.shape[0]))

        transform_types = ["diag", "upper_lower", "left_right", "mask"]

        for t in transform_types:
            if t == "mask":
                mask = add_salt_and_pepper_noise(mosaic_1.shape, amount=0.2)
            else:
                mask = np.zeros(())

            result_image = build_mosaics(mosaic_1, mosaic_2, type=t, mask=mask)
            result_label = build_mosaics(label_1, label_2, type=t, mask=mask)

            result_image_path = os.path.join(output_image_folder, f"{image_file.split('.')[0]}_{t}.jpg")
            result_label_path = os.path.join(output_label_folder, f"{image_file.split('.')[0]}_{t}.png")

            cv2.imwrite(result_image_path, result_image)
            cv2.imwrite(result_label_path, result_label)

            print(f"Saved {result_image_path} and {result_label_path}")


# 设置文件夹路径
image_folder = "F:/unet-pytorch-main/VOCdevkit1/VOC2007/JPEGImages"
label_folder = "F:/unet-pytorch-main/VOCdevkit1/VOC2007/Segmentation"
output_image_folder = "F:/unet-pytorch-main/VOCdevkit1/VOC2007/MosaicImages"
output_label_folder = "F:/unet-pytorch-main/VOCdevkit1/VOC2007/MosaicLabels"

process_images(image_folder, label_folder, output_image_folder, output_label_folder)
