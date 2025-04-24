import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import pickle
import glob

# Set up the ArUco dictionary and detector parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(aruco_dict, parameters)

# Real-world dimensions of the marker (in cm)
marker_length = 8.0  # 8x8 cm

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Path to images (update this path to your images folder)
image_paths = glob.glob("img0.png")  # or *.jpg if applicable

last_gray = None  # to store the grayscale image for later use

for image_path in image_paths:
    img = cv.imread(image_path)
    if img is None:
        continue  # skip if image couldn't be loaded
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    last_gray = gray  # update last_gray with current image's gray
    
    # Detect ArUco markers
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is not None:
        for corner in corners:
            # Define 3D object points for the marker (assume marker is centered at (0,0,0))
            objp = np.array([
                [-marker_length / 2,  marker_length / 2, 0],
                [ marker_length / 2,  marker_length / 2, 0],
                [ marker_length / 2, -marker_length / 2, 0],
                [-marker_length / 2, -marker_length / 2, 0]
            ], dtype=np.float32)
            objpoints.append(objp)
            imgpoints.append(corner[0])
        print(f"Processed image: {image_path}")
    else:
        print(f"No marker detected in: {image_path}")

if last_gray is None:
    print("No valid images found. Exiting.")
    exit()

if len(objpoints) < 5:
    print("Warning: Fewer than 5 frames captured. Calibration may be unreliable.")

# Use the shape of the last processed image for calibration
image_size = last_gray.shape[::-1]

# Camera calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
    objpoints, imgpoints, image_size, None, None
)

print("Camera calibration successful!")
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", dist_coeffs)

# Save calibration results
with open("cameraMatrix.pkl", "wb") as f:
    pickle.dump(camera_matrix, f)
with open("distCoeffs.pkl", "wb") as f:
    pickle.dump(dist_coeffs, f)

print("Calibration complete. Camera parameters saved.")
