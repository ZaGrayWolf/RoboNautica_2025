import numpy as np
import cv2
from cv2 import aruco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import Float32

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.subscription = self.create_subscription(
            Image,
            'usb_cam/image_raw',  # Ensure this matches your camera topic
            self.image_callback,
            10)
        
        self.distance_publisher = self.create_publisher(Float32, 'aruco_distance', 10)
        self.yaw_publisher = self.create_publisher(Float32, 'aruco_yaw', 10)
        self.bridge = CvBridge()

        # Load camera calibration data
        try:
            self.camera_matrix = np.load('/home/souri/robonautic_ws/cam_mtx.npy')
            self.dist_coeffs = np.load('/home/souri/robonautic_ws/dist.npy')
            self.get_logger().info("Loaded calibration files successfully.")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration files: {e}")
            return

        # Change to a more reliable dictionary if necessary
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)  # Changed from 1000 to 250
        self.parameters = aruco.DetectorParameters()
        self.marker_length = 0.1  # Increase if needed (adjust based on actual marker size)

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect ArUco markers
        detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is not None:
            self.get_logger().info(f"Detected ArUco IDs: {ids.flatten().tolist()}")
            aruco.drawDetectedMarkers(frame, corners, ids)  # Draw detected markers

            # Estimate pose of each marker
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

            for i in range(len(ids)):
                marker_id = ids[i][0]
                aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.1)

                # Calculate distance
                distance = np.linalg.norm(tvecs[i][0])
                self.distance_publisher.publish(Float32(data=distance))

                # Calculate yaw angle
                rmat, _ = cv2.Rodrigues(rvecs[i])
                yaw_angle = np.arctan2(rmat[1, 0], rmat[0, 0])
                self.yaw_publisher.publish(Float32(data=yaw_angle))

                # Display detected marker ID and distance on the frame
                center_x = int(np.mean(corners[i][0][:, 0]))
                center_y = int(np.mean(corners[i][0][:, 1]))
                cv2.putText(frame, f"ID: {marker_id}", (center_x, center_y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Dist: {distance:.2f}m", (center_x, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        else:
            self.get_logger().warn("No ArUco markers detected.")

        # Show the frame (if using remote SSH, use a local X11 display)
        cv2.imshow('Aruco Detection', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
