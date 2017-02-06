import cv2
import numpy as np

import config


# Use color transforms, gradients, etc., to create a thresholded binary image.

def pipeline(img):
    """
    Pipeline for a combined threshold pipeline,
    thresholds found from testing different options
    but also from guidance on slack
    :param img:
    :return:
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s_img = hls[:, :, 2]
    sobelx_on_simg = thresholded_sobel_x(s_img)
    sobelx_on_greyimg = thresholded_sobel_x(grey)
    r_thresh = rgb_binary_threshold_r(img)
    thresh = threshold(s_img, config.threshold_configs['s_img_min'], config.threshold_configs['s_img_max'])
    return combined(sobelx_on_greyimg, sobelx_on_simg, thresh, r_thresh)


def rgb_binary_threshold_r(src_img, thresh=(200, 255)):
    """Return a binary threshold image of the R channel in RGB color space."""
    R = src_img[:, :, 0]
    binary = np.zeros_like(R)
    binary[(R > thresh[0]) & (R <= thresh[1])] = 1
    return binary


def combined(sobel_x, sobelx_simg, s_thresh, r_thresh):
    """
    Return combined threshold
    :param sobel_x:
    :param sobelx_simg:
    :param s_thresh:
    :return:
    """
    combined = np.zeros_like(sobel_x, dtype=np.uint8)
    combined[(sobel_x > 0) | (sobelx_simg > 0) | (s_thresh > 0) | (r_thresh > 0)] = 255
    return combined


def thresholded_sobel_x(img):
    """
    Return a thresholded sobel
    :param img:
    :return:
    """
    abs_sobelx = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))  # Set kernal size?
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
    return threshold(scaled_sobel, config.threshold_configs['sobel_min'], config.threshold_configs['sobel_max'])


def threshold(image, thresh_min, thresh_max):
    """
    Generic threshold
    :param image:
    :param thresh_min:
    :param thresh_max:
    :return:
    """
    binary = np.zeros(image.shape, dtype=np.float32)
    binary[(image >= thresh_min) & (image <= thresh_max)] = 1
    return binary
