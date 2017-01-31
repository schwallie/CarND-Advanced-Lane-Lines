#Advanced Lane Finding Project

The goals / steps of this project are the following:

[X] Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
[X] Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[checkerboard]: ./camera_cal_final/checkerboard_undistorted.png "Checkerboard"
[undistorted]:  ./camera_cal_final/test_undistorted_plot.png "Undistorted"
[transformed]: ./output_images/persp_transform.png "Road Transformed"
[thresh]: ./output_images/threshold.png "Thresholded Image"
[diff]: ./output_images/diff_bw_lanes.png "Difference Between Lanes"
[final]: ./output_images/final.png "Output"
[image6]: ./output_images/example_output.jpg "Output"
[video1]: ./project_video_annotated.mp4 "Video"


###Camera Calibration

The code for calibration is conveniently located in `calibrate_camera.py`.  

I start by preparing "object points" (objpoints), which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Checkerboard][checkerboard]


###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

![Undistorted v Distorted][undistorted]

####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `get_warped_perspective()`, which appears in lines 7-14 in the file `persp_transform.py` (persp_transform.py)  The `get_warped_perspective()` function takes as inputs an image (`img`).  I chose the hardcode the source and destination points in the following manner:

src(w=width, h=height):
```
    top = math.floor(h * .72)
    close_left = math.floor(w * .15)
    close_right = math.floor(w * .86)
    far_left = math.floor(w * .37)
    far_right = math.floor(w * .63)
```

dest(w=width):
```
    top_left = w * .25
    topr_pct = .75
    top_right = w * topr_pct
```
This resulted in the following source and destination points:

```
SRC=[[  473.   532.]
 [  832.   532.]
 [  128.   719.]
 [ 1126.   719.]]
DEST=[[ 320.  540.]
 [ 960.  540.]
 [ 320.  720.]
 [ 960.  720.]]
```

![Transformed Image][transformed]

####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps in `thresholds.py`).  I learned a lot from Slack + Forums + others when figuring out what kind of thresholds to use for this particular part.  

Here's an example of my output for this step. 

![Binary Threshold Combo][thresh]


####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In `histogram_img_search.py` I used a sliding window up the image to identify peaks in the image histogram where lines were present. This built out a large list of points that were likely to be lines for my eventual fitting of the line.

In `draw_lanes.py` I used `numpy.polyfit` to fit the lines, and used `cv2.fillPoly()` to fill the lanes on the image

![alt text][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![Output][final]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

![Histogram of Widths of Lanes][diff]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
