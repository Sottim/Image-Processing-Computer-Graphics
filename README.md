## Panorama Stitching Project
Panoramic images provide a wide field of view, offering an immersive experience in photography. However, creating seamless panoramas from multiple overlapping images can be challenging due to the need for precise image alignment and seamless blending. This project aims to address these challenges by using algorithms to automatically detect overlapping regions and align the images correctly, utilizing techniques like feature detection and feature matching.

### Methodology

##### 1. SIFT Feature Detection
Detect keypoints and compute descriptors using SIFT for each input image. The algorithm used is the scale-invariant feature transform which helps to detect interest points, describe, and match local features in images. 

##### 2. Feature Matching
It is useful to match and establish the descriptors between overlapping image pairs to find correspondences. Keypoints are characterized by descriptors, which are feature vectors representing the local image information around each keypoint. Euclidean distance measures the similarity or dissimilarity between the descriptors of two keypoints. Smaller the distances higher is the similarity.

##### 3. Homography Estimation
The matched feature correspondences are used to estimate the homography matrix between image pairs. This matrix describes the geometric relationship between corresponding points in the two images. An specific algorithm will be used to find the values of this matrix such as Direct Linear Transformation or Normalized Direct Linear Transformation depending upon the noise in the image captured.

##### 4. Image Warping and Stitching
Apply the estimated homography to warp and align the overlapping images onto a common coordinate system, forming the panoramic view.

##### 5. Weighted Blending
Depending upon the output from the above steps weighted blending technique such as feathering will be applied. It will seamlessly merge the overlapping regions and minimize visible seams or exposure differences because simply averaging the images doesn't solve the issue and seams might still be visible. Thus to remove hard seams that may arise due to vignetting, exposure differences, blending technique will be used.

### Expected Outcomes
Development of a robust panorama stitching pipeline capable of handling varying image orientations and lighting conditions.
Stitching multiple images together to obtain a wider panorama. <br>
Additionally, I am to include an input image loading and selection interface as well as an interactive panorama viewer in this project. This will allow users to easily load and select the set of overlapping images they want to stitch into a panoroma and to zoom and navigate through the stitched panoramic image.

### Usage
Input: Multiple overlapping images. <br>
Output: A stitched panoramic image.

### Dependencies
OpenCV : Used for SIFT feature detection, feature matching, homography estimation, image warping, and blending. <br>
numpy : Used for numerical operations

#### Installation the dependencies using:    
    pip install opencv-python numpy

    Run the python script for panorama stitching where input images are arguments:
        python panorama_stitching.py image1.jpg image2.jpg ...

### Contributors
    * Santosh Adhikari


