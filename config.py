import cv2
# CAMERA CALIBRATION CONFIGS
win_size = (11, 11)
zero_zone = (-1, -1)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
calibrate_path='camera_cal/*'
calibrate_output='camera_cal_final/'
test_img = 'test_images/test1.jpg'