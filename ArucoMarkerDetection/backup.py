import cv2
import numpy as np
import pickle
from cv2 import aruco

def detect_aruco_markers(image_path=None, camera_id=0, use_camera=False, marker_length=0.05, calib_file='calibration.pkl'):
    """
    Detect ArUco markers in an image or camera feed, estimate their pose, and calculate the distance.
    
    Args:
        image_path (str, optional): Path to the input image.
        camera_id (int, optional): Camera ID for webcam capture.
        use_camera (bool, optional): If True, use webcam instead of image file.
        marker_length (float, optional): The length of the ArUco marker's side (in meters).
        calib_file (str, optional): Path to the pickle file containing camera calibration parameters.
        
    Returns:
        None: Displays the image with detected markers and their distances.
    """
    # Load camera calibration parameters from pickle file
    with open(calib_file, 'rb') as f:
        calib_data = pickle.load(f)
    
    # Check calibration data type
    if isinstance(calib_data, dict):
        camera_matrix = calib_data['mtx']
        dist_coeffs = calib_data['dist']
    elif isinstance(calib_data, (tuple, list)) and len(calib_data) >= 2:
        camera_matrix = calib_data[0]
        dist_coeffs = calib_data[1]
    else:
        print("Error: Calibration data format not recognized.")
        return

    # Set up the ArUco dictionary and detector parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    
    if use_camera:
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return
        print("Press 'q' to quit")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            process_frame(frame, detector, camera_matrix, dist_coeffs, marker_length)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        if image_path is None:
            print("Please provide an image path or set use_camera=True")
            return
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return
        process_frame(image, detector, camera_matrix, dist_coeffs, marker_length)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def process_frame(frame, detector, camera_matrix, dist_coeffs, marker_length):
    """
    Process a frame to detect, annotate ArUco markers, estimate their pose, and calculate distance.
    
    Args:
        frame: Input image frame.
        detector: ArUco detector object.
        camera_matrix: Camera matrix from calibration.
        dist_coeffs: Distortion coefficients from calibration.
        marker_length: The length of the ArUco marker's side (in meters).
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Try to use the built-in function if available
        if hasattr(aruco, "estimatePoseSingleMarkers"):
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)
            # Process each detected marker
            for i, marker_corners in enumerate(corners):
                marker_id = ids[i][0]
                # Draw the axis for each marker
                aruco.drawAxis(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], marker_length * 0.5)
                tvec = tvecs[i][0]
                distance = np.linalg.norm(tvec)
                center_x = int(np.mean(marker_corners[0][:, 0]))
                center_y = int(np.mean(marker_corners[0][:, 1]))
                cv2.putText(frame, f"ID: {marker_id}", (center_x, center_y - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(frame, f"Distance: {distance:.2f} m", (center_x, center_y - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Fallback: use cv2.solvePnP for each detected marker
            for i, marker_corners in enumerate(corners):
                marker_id = ids[i][0]
                img_points = marker_corners[0].astype(np.float32)
                # Define object points in the marker coordinate system (assuming the marker is centered at (0,0,0))
                obj_points = np.array([
                    [-marker_length / 2,  marker_length / 2, 0],
                    [ marker_length / 2,  marker_length / 2, 0],
                    [ marker_length / 2, -marker_length / 2, 0],
                    [-marker_length / 2, -marker_length / 2, 0]
                ], dtype=np.float32)
                ret, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
                if ret:
                    # Draw axis
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, marker_length * 0.5, 2)
                    distance = np.linalg.norm(tvec)
                    center_x = int(np.mean(marker_corners[0][:, 0]))
                    center_y = int(np.mean(marker_corners[0][:, 1]))
                    cv2.putText(frame, f"ID: {marker_id}", (center_x, center_y - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"Distance: {distance:.2f} m", (center_x, center_y - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
    cv2.imshow("ArUco Marker Detection", frame)

if __name__ == "__main__":
    detect_aruco_markers(use_camera=True)
    # For image file use:
    # detect_aruco_markers(image_path="path_to_your_image.jpg", use_camera=False)
