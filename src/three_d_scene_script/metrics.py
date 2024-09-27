import numpy as np
import cv2
from sklearn.metrics import mean_squared_error
from scipy.optimize import linear_sum_assignment
from lpips import LPIPS
import torch
import math


def calculate_psnr(gt_image, output_image):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the ground truth image and the output image.
    PSNR is used as a cost for Hungarian matching.
    """
    mse_value = np.mean((gt_image - output_image) ** 2)
    if mse_value == 0:
        return 0
    PIXEL_MAX = 255.0
    psnr_value = 20 * math.log10(PIXEL_MAX / math.sqrt(mse_value))
    return 100 - psnr_value


def calculate_mse(gt_image, output_image):
    """
    Calculate the Mean Squared Error (MSE) between the ground truth image and the output image.
    MSE is used as a cost for Hungarian matching.
    """
    mse_value = mean_squared_error(gt_image.flatten(), output_image.flatten())
    return mse_value


def calculate_lpips(gt_image, output_image):
    """
    Calculate the Learned Perceptual Image Patch Similarity (LPIPS) between the ground truth image and the output image.
    LPIPS is used as a cost for Hungarian matching.
    """
    lpips_model = LPIPS(net='vgg').cuda()
    gt_tensor = torch.from_numpy(gt_image).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
    output_tensor = torch.from_numpy(output_image).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
    lpips_value = lpips_model(gt_tensor, output_tensor).item()
    return lpips_value


def calculate_euclidean_distance(gt_image, output_image):
    """
    Calculate the Euclidean Distance between the ground truth image and the output image.
    Euclidean distance is used as a cost for Hungarian matching.
    """
    gt_image_flat = gt_image.flatten()
    output_image_flat = output_image.flatten()
    euclidean_dist = np.linalg.norm(gt_image_flat - output_image_flat)
    return euclidean_dist


def hungarian_matching(cost_matrix):
    """
    Perform Hungarian matching using the provided cost matrix.

    Args:
    cost_matrix (numpy.ndarray): A 2D array where each entry represents a cost.

    Returns:
    tuple: row and column indices of the optimal matching.
    """
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return row_ind, col_ind


def run_all_metrics(gt_image_path, output_image_path):
    """
    Run all the image quality metrics between the ground truth image and the output image,
    and apply Hungarian matching to minimize costs.

    Args:
    gt_image_path (str): The file path to the ground truth image.
    output_image_path (str): The file path to the output image.

    Returns:
    dict: A dictionary containing all the image quality metrics.
    """

    gt_image = cv2.imread(gt_image_path)
    output_image = cv2.imread(output_image_path)

    gt_image_gray = cv2.cvtColor(gt_image, cv2.COLOR_BGR2GRAY)
    output_image_gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)

    psnr_value = calculate_psnr(gt_image, output_image)
    mse_value = calculate_mse(gt_image, output_image)
    lpips_value = calculate_lpips(gt_image, output_image)
    euclidean_distance = calculate_euclidean_distance(gt_image, output_image)

    cost_matrix = np.array([
        [psnr_value, mse_value],
        [lpips_value, euclidean_distance]
    ])

    row_ind, col_ind = hungarian_matching(cost_matrix)

    return {
        "PSNR": psnr_value,
        "MSE": mse_value,
        "LPIPS": lpips_value,
        "Euclidean Distance": euclidean_distance,
        "Hungarian Matching": (row_ind, col_ind)
    }
