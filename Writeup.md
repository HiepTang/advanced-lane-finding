## Advanced Lane Finding Project | Writeup

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)
[chessboard_img]:            ./readme_images/chessboard.png                  "Chessboard"
[original_img]:              ./test_images/test3.jpg                         "Original"
[undistorted_img]:           ./output_images/test3_undistorted.jpg           "Undistorted"
[thresholded_img]:           ./output_images/test3_thresholded.jpg           "Thresholded"
[reg_of_interest_img]:       ./output_images/test3_region_of_interest.jpg    "Reg. of Interest"
[warped_img]:                ./output_images/test3_warped.jpg                "Warped"
[first_fit_polynomial_img]:  ./output_images/test3_first_fit_polynomial.jpg  "Sliding Window Search"
[second_fit_polynomial_img]: ./output_images/test3_second_fit_polynomial.jpg "Skip Sliding Window Search"
[polygon_img]:               ./output_images/test3_polygon.jpg               "Polygon"
[text_img]:                  ./output_images/test3_text.jpg                  "Curv. Radius and Bias"
[video]:                    ./output_videos/project_video.mp4               "Final Video"

### Description
Code responsible for lane lines detection is placed in [advanced_lane_finding.py](./advanced_lane_finding.py) python module. Jupyter notebook [Advanced_Lane_Finding.ipynb](./Advanced_Lane_Finding.ipynb) contains the code to illustrate the result of each step of the lane lines detecting pipeline. Calibration images are placed in [chessboard_images](./chessboard_images) directory and the test images are placed in [test_images](./test_images) directory. Input video is located in [input_videos](./input_videos) directory. Output images, illustrating how pipeline works, and the result video with the detected lane boundaries are placed in [output_images](./output_images) and [output_videos](./output_videos) correspondingly.

Pipeline steps will be illustrated using the following input image:

![alt text][original_img]

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

#### Writeup / README

##### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

Project has both: [Writeup.md](./Writeup.md) and [README.md](./README.md).

#### Camera Calibration

##### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is can be found in `AdvancedLaneFinder::calibrate_camera` in [advanced_lane_finding.py](./advanced_lane_finding.py).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. An example of undistorted image generated using camera calbration results is shown below.

![alt text][chessboard_img]

#### Pipeline (Images)

##### 1. Provide an example of a distortion-corrected image.

Here is an example of undistorted image. Distortion correction is implemented in `AdvancedLaneFinder::distortion_correction` in [advanced_lane_finding.py](./advanced_lane_finding.py). The difference with the original image can be seen by looking at the trees and bottom left and right edges.

![alt text][undistorted_img]

##### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps are in `AdvancedLaneFinder::apply_thresholds` in [advanced_lane_finding.py](./advanced_lane_finding.py) which subsequently calls `AdvancedLaneFinder::_abs_threshold`, `AdvancedLaneFinder::_mag_threshold`, `AdvancedLaneFinder::_dir_threshold`, and `AdvancedLaneFinder::_s_channel_threshold`.  Here's an example of my output for this step.

![alt text][thresholded_img]

I have also used region of interest technique to get rid of trees, mountains, etc.

![alt text][reg_of_interest_img]

##### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in `AdvancedLaneFinder::warp_perspective`.  Perspective transform matrix is intialized in `AdvancedLaneFinder::__init__`. Source and destination points are provided as arguments in constructor. They were extracted from one of the test images presenting straight lines. Here are they:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 532,  496     | 288,  366     | 
| 756,  496     | 1016, 366     |
| 288,  664     | 288,  664     |
| 1016, 664     | 1016, 664     |

The example of the perspective transformation is given below.

![alt text][warped_img]

##### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

To find the points to be fitted with the second-order polynomial I've used "sliding window" technique. For the rest of images "sliding window" was skipped because it was already known were the points possibly are. The code can be found in `AdvancedLaneFinder::fit_polynomial`, `AdvancedLaneFinder::_sliding_window_fit`, and `AdvancedLaneFinder::_skip_sliding_window_fit`. Here are illustrations of sliding window seach and the search technique using margins around previously fitted lines. Also, a simple sanity test was implemented that prevents catastrofic misdetections. The difference between bottom pixels of the new and old fitted lines is calculated and, if it is greater than predefined constant, old fitted coordinates are left unchanged. If certain number (another constant) of failures happen in a row, then sliding window search is performed again.

![alt text][first_fit_polynomial_img]

![alt text][second_fit_polynomial_img]

##### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Curvature radius and bias fron center were calculated in `AdvancedLaneFinder::_calculate_curvature` and `AdvancedLaneFinder::_calculate_bias_from_center` correspondingly. The text was added to the image in the last step of pipeline.

![alt text][text_img]

##### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This step is implemented in `AdvancedLaneFinder::draw_polygon` in [advanced_lane_finding.py](./advanced_lane_finding.py).

![alt text][polygon_img]


#### Pipeline (Video)

##### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

The result video can be found in [this repository](./output_videos/project_video.mp4) as well as on [YouTube](https://www.youtube.com/watch?v=iRGHCgbpnOk).


#### Discussion

##### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In this project I followed the recomendations given by Udacity. The trickiest part was to create a reusable python module. It was also hard to implement sliding window technique and its visualization.

The implementation performs well on the project video and not so well on other videos with worse environment conditions. The pipeline will likely fail in the situations when the road has cracks coming alongside the lane lines, such as in challenge video. Sanity checks and recovery are helping here---eventually the lanes are detected correctly, but there are noticable sequence of frames on the chanllenge video, when lines are detected completely wrong. 

Smothing and failure recovery should be more sophisticated to make `AdvancedLaneFinder` work with videos capturing worse environment conditions. Lane detection can be improved with the follwoing approach: obtain lane pixels by color. The rationale behind it is that lanes in this project are either yellow or white. 

