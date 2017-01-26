import cv2

import config
import persp_transform
import thresholds

class Pipeline(object):
    def __init__(self, override_calibration=False, test_img='test_images/straight_lines1.jpg'):
        if not override_calibration:
            self.mtx = config.mtx
            self.dist = config.dist
        else:
            import calibrate_camera
            # Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
            c = calibrate_camera.Calibrate()
            self.mtx, self.dist = c.calibrate()
        self.test_img = test_img

    def main(self, img=None):
        """
        Use color transforms, gradients, etc., to create a thresholded binary image.
        Apply a perspective transform to rectify binary image ("birds-eye view").
        Detect lane pixels and fit to find the lane boundary.
        Determine the curvature of the lane and vehicle position with respect to center.
        Warp the detected lane boundaries back onto the original image.
        Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

        From @kylesf on Slack:
        [X] Camera Calibration
        [X] Distortion Correction
        [X] Perspective Transform
        4. Color and Gradient Threshold
        5. Detect Lane Lines
        6. Determine Lane Curvature
        7. Impose Lane Boundaries on Original Image
        8. Output Visual Display of Lane Boundaries and Numerical Estimation of Lane Curvature and Vehicle Position

        From Kostas:
        - No lane peaks from histogram:
          * first frame : skip frame
          * other frame:  get previous good frame / count bad frame data PER LANE

        - One or more peaks are missing: (almost same as above)
           * first frame : not applicable because we cannot compare with previous state
           * other frame :
                     a) increase the count for bad frames for the current lane
                     b) if we have reached a critical count (say 10 lost frames) we search again for peaks. if it is still missing we stop drawing that lane
                     c) if not we bring the lane polynomial from the previous frame and check if it is compatible with the other lanes (discuss that below)

        - One or more peaks is found but has curvature that is out of bounds or much different from the other lanes:
            a) we can either pick from the previous good frame (if there is one)
             or b) copy from the other lanes by offsetting the curve
            It is also important which lane is it. As we go to the right , its less important (if we monitor more than one lane)


        - Check the first two lanes if they are still parallel (in the bottom of the image).
            If the lanes open up (i think we go uphil) and we need to devise an algorithm to change perceptive to make them again parallel

        so in general
          a) copy from previous frame
          b) copy from other lanes
          c) if exceeds number of bad frames search the histogram again and if it fails drop the lane
          d) check for perspective changes

        First quantity (ensure i have data, if not get from past frame), then quality (ensure they are compatible, first with each other then with previous frame)
        :return:
        """
        if img is None:
            img = cv2.imread(self.test_img)
        # Apply a distortion correction to raw images.
        undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        # Perspective Transform
        persp = persp_transform.get_warped_perspective(undistorted)
        thresh = thresholds.thresh_pipeline(persp)
        return persp, thresh


if __name__ == '__main__':
    p = Pipeline()
    p.main()
