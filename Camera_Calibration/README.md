# Custom_ArucoMarker_Detection and Pose Estimation


This project demonstrates the detection of ArUco markers in a video feed and estimates their pose using a pre-calibrated camera. The detected markers are displayed with overlaid 3D coordinate axes, and their pose is computed using a custom Direct Linear Transform (DLT) method.

## Features

- **Marker Detection**: Detects ArUco markers in real-time.
- **Pose Estimation**: Estimates the 6-DOF pose (rotation and translation) of the markers.
- **Camera Calibration**: Uses pre-calibrated camera parameters for accurate pose estimation.
- **Marker Analysis**: Divides the marker image into cells and analyzes black/white pixels to compute marker bits.

## Prerequisites

- Python 3.x
- OpenCV (with ArUco and camera modules)
- Numpy
- Pickle (for loading camera parameters)

You can install the required libraries using pip:

```bash
pip install opencv-python opencv-contrib-python numpy

.
├── Camera_Calibration/
│   ├── cameraMatrix.pkl   # Calibrated camera matrix
│   ├── dist.pkl           # Camera distortion coefficients
├── README.md
├── aruco_pose_estimation.py

Usage
Camera Calibration: Ensure that the camera has been calibrated beforehand. The calibration files cameraMatrix.pkl and dist.pkl should be stored in the Camera_Calibration/ directory.

Run the Detection:

To run the marker detection and pose estimation, use the following command:
bash
python aruco_pose_estimation.py
The script will start capturing video from the default camera (/dev/video0 on Linux, the first available camera on Windows). Detected markers will be highlighted, and their pose will be displayed.

Quit the Application: Press q to exit the video feed.

Key Functions
detect_markers(image): Detects ArUco markers in the given image.
divide_image_into_cells(image, num_rows, num_cols): Divides the marker image into grid cells for bit analysis.
count_black_white_pixels(cell): Analyzes each cell for black and white pixel counts.
pnp_dlt(objPoints, imgPoints, cameraMatrix, distCoeff): Custom implementation of the PnP algorithm for pose estimation using Direct Linear Transform.
aruco_display(corners, ids, image): Overlays 3D axes on detected markers and displays their pose.
Camera Calibration
If you do not have the camera calibration files (cameraMatrix.pkl and dist.pkl), you will need to calibrate your camera first. You can use the OpenCV calibrateCamera() function to obtain these parameters by capturing images of a chessboard pattern.

License
This project is licensed under the MIT License - see the LICENSE file for details.



### Notes:
- This README assumes that the script file is named `aruco_pose_estimation.py`.
- Add instructions for camera calibration if required or link to a tutorial for calibrate
