# RoboNautica_2025

## Autonomous Robot Follow-the-Leader System

## 1. Problem Statement

The goal of this project is to enable a robot to autonomously follow a leader robot. This is achieved by using a marker-based tracking system, specifically ArUco markers, to allow the follower robot to maintain a constant distance and track the leader's movements[cite: 2]. The system must address key challenges such as leader detection, maintaining a desired following distance, dynamically adjusting its path to follow the leader, and avoiding collisions[cite: 2].

## 2. Method Used

We implemented an ArUco marker-based tracking system within the ROS2 framework. This approach involves the following:

* **ArUco Marker Detection:** OpenCV is used to detect ArUco markers attached to the leader robot. The follower robot's camera captures images, and the system processes these images to identify the markers and determine the leader's pose (position and orientation).
* **Pose Estimation:** By analyzing the marker's position in the camera frame, we estimate the leader robot's pose relative to the follower robot. This involves calculating both the translational and rotational vectors.
* **Control System:** A control system, potentially a PZD controller, is used to process the pose information and generate the necessary motor commands for the follower robot to maintain the desired following distance and match the leader's movements.
* **ROS2 Communication:** ROS2 is used for communication between different software components of the system, such as camera input, marker detection, control algorithms, and motor control.

## 3. Implementation Details

The implementation of the autonomous follow-the-leader system involves several key steps:

1.  **Marker Setup:** ArUco markers are attached to the leader robot in a configuration that allows for robust detection and pose estimation from various angles.
2.  **Camera Calibration:** The camera of the follower robot is calibrated to obtain intrinsic parameters (focal length, etc.) necessary for accurate pose estimation.
3.  **Marker Detection and Pose Estimation:**
    * The follower robot's camera captures images of the environment.
    * OpenCV functions are used to detect ArUco markers in the images.
    * For each detected marker, the system calculates the translation and rotation vectors, representing the leader's pose relative to the follower.
4.  **Control Algorithm:**
    * The pose information is fed into a control algorithm (e.g., a PZD controller).
    * The controller calculates the required linear and angular velocities for the follower robot to:
        * Maintain the desired distance from the leader.
        * Align its orientation with the leader.
        * Smoothly adjust its path as the leader moves.
5.  **Robot Motion Control:**
    * The calculated velocity commands are sent to the robot's motor controllers through ROS2 messages.
    * The robot's motors actuate to achieve the desired motion, enabling it to follow the leader[cite: 3].
6.  **Obstacle Avoidance:**
    * LIDAR (and potentially other proximity sensors) are used to detect obstacles in the follower robot's path[cite: 4].
    * The control algorithm is augmented to incorporate obstacle avoidance, ensuring the follower robot navigates safely while following the leader.

## 4. System Dependencies

* ROS2 (Robot Operating System 2)
* OpenCV
* aruco\_ros (ROS2 package for ArUco marker detection)
* (Other ROS2 packages for robot control, if applicable)

## 5. How to Run

1.  **Install ROS2:** Follow the ROS2 installation instructions for your operating system.
2.  **Install Dependencies:**
    ```bash
    sudo apt install ... # Example: ROS2 packages, OpenCV
    pip install ...   # Python dependencies
    ```
3.  **Clone the Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
4.  **Build the ROS2 Package:**
    ```bash
    colcon build
    ```
5.  **Source the ROS2 Environment:**
    ```bash
    source install/setup.bash
    ```
6.  **Run the Launch File:**
    ```bash
    ros2 launch <package_name> <launch_file_name>.launch.py
    ```

## 6. Notes

* Ensure proper calibration of the robot's camera for accurate pose estimation.
* The performance of the system may be affected by factors such as lighting conditions, marker visibility, and robot dynamics.
* Further improvements can be made by incorporating more advanced control algorithms, sensor fusion techniques, and dynamic path planning.
