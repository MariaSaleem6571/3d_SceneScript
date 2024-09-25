# metrics.py
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
from scipy.optimize import linear_sum_assignment
from lpips import LPIPS
import torch
import math

def calculate_ssim(gt_image, output_image):
    """
    Calculate the Structural Similarity Index (SSIM) between the ground truth image and the output image.

    Args:
    gt_image (numpy.ndarray): The ground truth image.
    output_image (numpy.ndarray): The output image.

    Returns:
    float: The SSIM value.
    """

    ssim_value = ssim(gt_image, output_image, multichannel=True)
    print(f'SSIM: {ssim_value}')
    return ssim_value

def calculate_psnr(gt_image, output_image):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between the ground truth image and the output image.

    Args:
    gt_image (numpy.ndarray): The ground truth image.
    output_image (numpy.ndarray): The output image.

    Returns:
    float: The PSNR value.
    """

    mse_value = np.mean((gt_image - output_image) ** 2)
    if mse_value == 0:
        return 100
    PIXEL_MAX = 255.0
    psnr_value = 20 * math.log10(PIXEL_MAX / math.sqrt(mse_value))
    print(f'PSNR: {psnr_value}')
    return psnr_value

def calculate_mse(gt_image, output_image):
    """
    Calculate the Mean Squared Error (MSE) between the ground truth image and the output image.

    Args:
    gt_image (numpy.ndarray): The ground truth image.
    output_image (numpy.ndarray): The output image.

    Returns:
    float: The MSE value.
    """

    mse_value = mean_squared_error(gt_image.flatten(), output_image.flatten())
    print(f'MSE: {mse_value}')
    return mse_value

def calculate_lpips(gt_image, output_image):
    """
    Calculate the Learned Perceptual Image Patch Similarity (LPIPS) between the ground truth image and the output image.

    Args:
    gt_image (numpy.ndarray): The ground truth image.
    output_image (numpy.ndarray): The output image.

    Returns:
    float: The LPIPS value.
    """

    lpips_model = LPIPS(net='vgg').cuda()
    gt_tensor = torch.from_numpy(gt_image).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
    output_tensor = torch.from_numpy(output_image).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0
    lpips_value = lpips_model(gt_tensor, output_tensor).item()
    print(f'LPIPS: {lpips_value}')
    return lpips_value

def calculate_euclidean_distance(gt_image, output_image):
    """
    Calculate the Euclidean Distance between the ground truth image and the output image.

    Args:
    gt_image (numpy.ndarray): The ground truth image.
    output_image (numpy.ndarray): The output image.

    Returns:
    float: The Euclidean Distance.
    """

    gt_image_flat = gt_image.flatten()
    output_image_flat = output_image.flatten()
    euclidean_dist = np.linalg.norm(gt_image_flat - output_image_flat)
    print(f'Euclidean Distance: {euclidean_dist}')
    return euclidean_dist

def run_all_metrics(gt_image_path, output_image_path):
    """
    Run all the image quality metrics between the ground truth image and the output image.

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

    ssim_value = calculate_ssim(gt_image_gray, output_image_gray)
    psnr_value = calculate_psnr(gt_image, output_image)
    mse_value = calculate_mse(gt_image, output_image)
    lpips_value = calculate_lpips(gt_image, output_image)
    euclidean_distance = calculate_euclidean_distance(gt_image, output_image)

    return {
        "SSIM": ssim_value,
        "PSNR": psnr_value,
        "MSE": mse_value,
        "LPIPS": lpips_value,
        "Euclidean Distance": euclidean_distance,
    }
