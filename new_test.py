import cv2
import numpy as np


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


def test_upper_lower():
    # 使用您提供的路径加载两张图像
    mosaic_1 = cv2.imread("F:/unet-pytorch-main/VOCdevkit1/VOC2007/JPEGImages/000201.jpg", cv2.IMREAD_GRAYSCALE)
    mosaic_2 = cv2.imread("F:/unet-pytorch-main/VOCdevkit1/VOC2007/JPEGImages/000550.jpg", cv2.IMREAD_GRAYSCALE)

    # 检查是否成功加载
    if mosaic_1 is None or mosaic_2 is None:
        raise FileNotFoundError("无法读取图像，请检查路径是否正确")

    # 确保两张图像大小相同
    if mosaic_1.shape != mosaic_2.shape:
        mosaic_2 = cv2.resize(mosaic_2, (mosaic_1.shape[1], mosaic_1.shape[0]))

    # 生成拼接图像（使用 upper_lower 模式）
    result_image = build_mosaics(mosaic_1, mosaic_2, type="upper_lower")

    # 显示结果
    cv2.imshow("Upper Lower Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 运行测试函数
test_upper_lower()
