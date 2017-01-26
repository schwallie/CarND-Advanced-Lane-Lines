import cv2
import numpy as np
import math
import config


def get_warped_perspective(img):
    w = img.shape[1]
    h = img.shape[0]
    src = get_src(w, h)
    dest = get_dest(w, h)
    matrix = cv2.getPerspectiveTransform(src, dest)
    flipped = img.shape[0:2][::-1]
    return cv2.warpPerspective(img, matrix, flipped)


def get_src(w, h):
    """
    Coordinates of quadrangle vertices in the source image.
    :param w:
    :param h:
    :return:
    """
    top = math.floor(h * config.perspect_configs['top'])
    close_left = math.floor(w * config.perspect_configs['close_left'])
    close_right = math.floor(w * config.perspect_configs['close_right'])
    far_left = math.floor(w * config.perspect_configs['far_left'])
    far_right = math.floor(w * config.perspect_configs['far_right'])
    return np.float32([(far_left, top), (far_right, top), (close_left, h-1), (close_right, h-1)])


def get_dest(w, h):
    """
    Coordinates of the corresponding quadrangle vertices in the destination image.
    :param w:
    :param h:
    :return:
    """
    top_left = w * config.perspect_configs['four_corners_top_left']
    topr_pct = config.perspect_configs['four_corners_top_right']
    top_right = w * topr_pct
    return np.float32(
        [[top_left, h * topr_pct],
         [top_right, h * topr_pct],
         [top_left, h],
         [top_right, h]])