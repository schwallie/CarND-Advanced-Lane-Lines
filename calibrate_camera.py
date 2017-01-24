import glob
import os

import cv2
import numpy as np

import config


class Calibrate(object):
    def __init__(self, save_images=True, plot_images=True):
        """
        Class for calibrating a camera to get rid of distortions
        :param save_images:
        """
        self.save_images = save_images
        self.path = config.calibrate_path
        self.output = config.calibrate_output
        self.img_paths = glob.glob(self.path)
        self.plot_images = plot_images

    def calibrate(self):
        """
        Main function to run the entire process
        :return:
        """
        if not os.path.exists(self.output):
            os.mkdir(self.output)
        objpoints = []
        imgpoints = []
        # Get the corners of all the  calibration images
        for img_path in self.img_paths:
            print(img_path)
            fname = os.path.basename(img_path)
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, objp, corners, nx, ny = self.find_chess_corners(gray)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                write_path = os.path.join(self.output, fname)
            else:
                write_path = os.path.join(self.output, '{0}_fail.jpg'.format(fname.split('.')[0]))
            if self.save_images:
                self.draw_and_save(img, nx, ny, corners, ret, write_path=write_path)
        # Use cv2's calibrateCamera to calibrate all the given points
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print('MTX={0} \n DIST={1}'.format(mtx, dist))
        if self.save_images:
            for img_path in self.img_paths:
                fname = os.path.basename(img_path)
                img = cv2.imread(img_path)
                undistorted = cv2.undistort(img, mtx, dist, None, mtx)
                cv2.imwrite(os.path.join(self.output, "{0}_undistorted.jpg".format(fname.split('.')[0])), undistorted)
            img = cv2.imread(config.test_img)
            undistorted = cv2.undistort(img, mtx, dist, None, mtx)
            undistorted_img_path = os.path.join(self.output, "test_img_undistorted.jpg")
            cv2.imwrite(undistorted_img_path, undistorted)
            if self.plot_images:
                import matplotlib.pyplot as plt
                f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
                f.tight_layout()
                ax1.imshow(img)
                ax1.set_title('Original Image', fontsize=50)
                ax2.imshow(undistorted)
                ax2.set_title('Undistorted Image', fontsize=50)
                plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
                # plt.show()
                self.undistored = undistorted
                self.original = img
        return mtx, dist

    @staticmethod
    def find_chess_corners(gray):
        """
        Finds the corners of the image.
        Because some images may be cut off, it counts down
        from 9,9 to 3,3 to find the image corners
        :param gray:
        :return:
        """
        for nx in range(9, 3, -1):
            for ny in range(9, 3, -1):
                ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
                print('({0}, {1}), {2}'.format(nx, ny, ret))
                if ret:
                    objp = np.zeros((nx * ny, 3), np.float32)
                    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
                    # http://docs.opencv.org/2.4/doc/tutorials/features2d/trackingmotion/corner_subpixeles/corner_subpixeles.html
                    # Use the OpenCV function cornerSubPix to find more exact corner positions.
                    # edits corners in place
                    cv2.cornerSubPix(gray, corners, winSize=config.win_size, zeroZone=config.zero_zone,
                                     criteria=config.criteria)
                    return ret, objp, corners, nx, ny
        return False, None, None, None, None

    def draw_and_save(self, img, nx, ny, corners, ret, write_path):
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        cv2.imwrite(write_path, img)


if __name__ == '__main__':
    c = Calibrate()
    mtx, dist = c.calibrate()
