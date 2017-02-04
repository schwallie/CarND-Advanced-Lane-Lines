import cv2
import numpy as np

import config


def get_window_for_lane(binary_warped, last_good_lane=None):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] / 2:, :], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    if last_good_lane and (
                    abs(leftx_base - rightx_base) > config.lane_configs['max_width'] or abs(leftx_base - rightx_base) <
                config.lane_configs['min_width']):
        print(
            'Overruling base. Old Left, Right = {0}, {1}, New L, R = {2}'.format(leftx_base, rightx_base,
                                                                                 last_good_lane))
        leftx_base, rightx_base = last_good_lane[0], last_good_lane[1]

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_size = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current_lst = [leftx_base]
    rightx_current_lst = [rightx_base]
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        leftx_current = leftx_current_lst[-1]
        rightx_current = rightx_current_lst[-1]
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_size
        win_y_high = binary_warped.shape[0] - window * window_size
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # print([(win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0)])
        # cv2.imwrite('win_full_{0}.png'.format(window), binary_warped[win_y_low:win_y_high, :])
        # cv2.imwrite('win_{0}.png'.format(window), binary_warped[win_y_low:win_y_high, win_xleft_low:win_xleft_high])
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix: # and window < 8:
            # print(window)
            # print(good_left_inds)
            # print('leftx_current_b4 = {0}'.format(leftx_current))
            leftx_current_lst.append(np.int(np.mean(nonzerox[good_left_inds])))
            # print('leftx_current_after = {0}'.format(leftx_current))
        if len(good_right_inds) > minpix: # and window < 8:
            rightx_current_lst.append(np.int(np.mean(nonzerox[good_right_inds])))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_curverad, right_curverad = get_center_radius(lefty, leftx, righty, rightx)
    return out_img, leftx, lefty, rightx, righty, leftx_base, rightx_base, left_curverad, right_curverad, leftx_current_lst, rightx_current_lst


def get_center_radius(lefty, leftx, righty, rightx):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30./720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meteres per pixel in x dimension
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*np.max(lefty) + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * np.max(lefty) + right_fit_cr[1]) ** 2) ** 1.5) \
                     / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters
    # Example values: 632.1 m    626.2 m
    return left_curverad, right_curverad