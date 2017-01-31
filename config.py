import cv2
import numpy as np

# CAMERA CALIBRATION CONFIGS
win_size = (11, 11)
zero_zone = (-1, -1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
calibrate_path = 'camera_cal/*'
calibrate_output = 'camera_cal_final/'
test_img = 'test_images/test1.jpg'
mtx = np.array([[1161.4544214947468, 0.0, 663.6021039294545],
                [0.0, 1158.838389558496, 389.1425902466672],
                [0.0, 0.0, 1.0]])
dist = np.array([[-0.2529169436992288, 0.03053774528012267,
                  -0.00023396495669634497, -0.00024374058563359847,
                  -0.10159488679195793]])

# Perspective Transform Configs:
perspect_configs = {'far_left': .37,
                    'close_left': .15,
                    'far_right': .63,
                    'close_right': .86,
                    'top': .72,
                    'four_corners_top_left': .25,
                    'four_corners_top_right': .75}

# Threshold Configs
threshold_configs = {'sobel_min': 20,
                     'sobel_max': 100,
                     's_img_min': 140,
                     's_img_max': 256}

# Histogram Image Search Configs
hist_configs = {'nwindows': 9,
                }

# Video Config
lane_configs = {'min_width': 400,
              'max_width': 650}