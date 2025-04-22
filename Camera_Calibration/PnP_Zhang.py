import cv2
import numpy as np
import pickle

with open('Camera_Calibration/cameraMatrix.pkl', 'rb') as f:
    cameraMatrix = pickle.load(f)
with open('Camera_Calibration/dist.pkl', 'rb') as f:
    distCoeffs = pickle.load(f)
    
# Check if camera parameters are read successfully
if cameraMatrix is None or distCoeffs is None:
    print("Invalid Camera File")
else:
    print("ALL GOOD")


ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def divide_image_into_cells(image, num_rows, num_cols):
    height, width = image.shape[:2]
    cell_height = height // num_rows
    cell_width = width // num_cols
    cells = []
    for i in range(num_rows):
        for j in range(num_cols):
            x = j * cell_width
            y = i * cell_height
            cell = image[y:y+cell_height, x:x+cell_width]
            cells.append(cell)
    return cells



def count_black_white_pixels(cell):
    # Ensure that the input cell is single-channel (grayscale)
    if len(cell.shape) > 2:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's method for binarization
    _, binary_cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Count black and white pixels
    num_black_pixels = np.sum(binary_cell == 0)
    num_white_pixels = np.sum(binary_cell == 255)
    return num_black_pixels, num_white_pixels


def analyze_bits(cells):
    bits = []
    for cell in cells:
        num_black_pixels, num_white_pixels = count_black_white_pixels(cell)
        bit = 0 if num_black_pixels > num_white_pixels else 1
        bits.append(bit)
    return bits


def detect_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    corners, regions, ids = [], [], []
    for contour in contours:
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Check if the contour has 4 vertices (i.e., a rectangle)
            x, y, w, h = cv2.boundingRect(approx)
            if w / h > 0.9 and w / h < 1.1:  # Aspect ratio close to 1 (i.e., approximately square)
                if cv2.contourArea(contour) > 500 and cv2.contourArea(contour) < 5000:
                    corners.append(approx.reshape(-1, 2))
                    region = (x, y, w, h)
                    regions.append(region)
                    marker_region = thresh[y:y+h, x:x+w]

                    # Analyze marker bits
                    cells = divide_image_into_cells(marker_region, 5, 5)  # Assuming 7x7 marker size
                    bits = analyze_bits(cells)
                    #print("Marker ID:", bits)
                    
                    # For demonstration purposes, just appending the sum of bits as an ID
                    ids.append(sum(bits))

    return corners, regions, ids


def undistort_points(camera_matrix, distortion_coeffs, image_points):
    
    """
    Undistorts a set of 2D image points using camera matrix and distortion coefficients.

    Args:
    - camera_matrix: Camera intrinsic matrix (3x3).
    - distortion_coeffs: Distortion coefficients (1x5 or 1x8).
    - image_points: Array of 2D image points to be undistorted (Nx2).

    Returns:
    - undistorted_points: Array of undistorted 2D image points (Nx2).
    """

    # Convert image points to numpy array if not already
    image_points_np = np.array(image_points, dtype=np.float32)
    print("Check1")
    # Extract camera parameters
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    # Extracting distortion coefficients from the nested list
    distortion_coeffs = (3.48533793e+00, 8.70974618e+01, 2.37235072e-01, -1.36174455e-01, 9.91264335e+02)

    # Distortion coefficients
    k1, k2, p1, p2, k3 = distortion_coeffs
   
    # Undistort each point
    undistorted_points = []
    for x, y in image_points_np:
        # Normalize image coordinates
        xn = (x - cx) / fx
        yn = (y - cy) / fy
        print("Check2")

        # Initial guess for undistorted coordinates
        x_u = xn
        y_u = yn

        # Iteratively apply distortion model to refine undistorted coordinates
        for _ in range(3):  # Iterate a few times for convergence
            r2 = x_u**2 + y_u**2
            k_radial = 1 + k1 * r2 + k2 * r2**2 + k3 * r2**3
            dx = 2 * p1 * x_u * y_u + p2 * (r2 + 2 * x_u**2)
            dy = p1 * (r2 + 2 * y_u**2) + 2 * p2 * x_u * y_u
            x_u = (xn - dx) / k_radial
            y_u = (yn - dy) / k_radial

        # Denormalize undistorted coordinates
        x_u = x_u * fx + cx
        y_u = y_u * fy + cy

        undistorted_points.append([x_u, y_u])
    
    print("Check3")
    print(undistorted_points)

    return np.array(undistorted_points)

    

def pnp_dlt(objPoints, imgPoints, cameraMatrix, distCoeff):
    """
    Perspective-n-Point (PnP) algorithm using Direct Linear Transform (DLT) method.

    Args:
    - objPoints: Array of 3D object points in the object coordinate space (Nx3).
    - imgPoints: Array of 2D image points (Nx2).
    - cameraMatrix: Camera intrinsic matrix (3x3).
    - distCoeff: Distortion coefficients (1x5 or 1x8).

    Returns:
    - rvec: Rotation vector (1x3).
    - tvec: Translation vector (1x3).
    """
    print("Hello_WWW")
    markerLength = 5.0
    # Undistort image points
    undistorted_imgPoints = undistort_points(cameraMatrix, distCoeff, imgPoints)
    print(undistorted_imgPoints)
    #und=cv2.undistortPoints(cameraMatrix, distCoeff, imgPoints)
    print("Undistort to objPoints")
    # Convert object points to homogeneous coordinates
    objPoints_hom = np.hstack((objPoints, np.ones((objPoints.shape[0], 1), dtype=np.float32)))
    print(objPoints_hom)
    # Direct Linear Transform (DLT) method
    A = []
    for i in range(objPoints.shape[0]):
        print("hello_DLT")
        X, Y, Z, _ = objPoints_hom[i, :]
        u, v = undistorted_imgPoints[i, :]
        A.append([X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u])
        A.append([0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v])

    A = np.array(A, dtype=np.float32)

    # Perform Singular Value Decomposition (SVD)
    _, _, V = np.linalg.svd(A)
    print("Hello_SVD")
    # Extract camera pose from the right nullspace vector
    P = V[-1, :].reshape((3, 4))

    # Estimate rotation and translation vectors
    R = P[:, :3]
    t = P[:, 3]    

    print("R",R)
    # Perform iterative refinement (optional)
    for _ in range(3):  # Perform 3 iterations
        markerLength_i = markerLength * 1.5
        print("+_+")
        for i in range(R.shape[0]):  # Iterate over each rotation vector
                    # Extract rotation vector from the rotation matrix
            _, rvec = cv2.Rodrigues(R[i].astype(np.float32))

            # Define axis points for projection
            axisPoints = np.float32([[0, 0, 0], [markerLength_i, 0, 0], [0, markerLength_i, 0], [0, 0, -markerLength_i]])

            # Project axis points onto the image plane using current pose estimation
            projected_imgPoints, _ = cv2.projectPoints(axisPoints.astype(np.float32), rvec, t[i].astype(np.float32), cameraMatrix.astype(np.float32), distCoeff.astype(np.float32))
            print("Projected_imgPoints: ", projected_imgPoints)

            # Print axisPoints shape here
            print("axisPoints shape:", axisPoints.shape)

            # Compute error between projected and actual image points
            error = undistorted_imgPoints - projected_imgPoints.squeeze()

            # Compute Jacobian matrix
            J = np.zeros((2 * objPoints.shape[0], 6), dtype=np.float32)

            for j in range(objPoints.shape[0]):
                X, Y, Z, _ = objPoints_hom[j, :]
                u, v = projected_imgPoints[j, 0], projected_imgPoints[j, 1]
                J[2 * j, :] = [-X/Z, -Y/Z, -1/Z, 0, 0, 0]
                J[2 * j + 1, :] = [0, 0, 0, -X/Z, -Y/Z, -1/Z]

            # Compute update step using pseudo-inverse
            delta = np.linalg.lstsq(J, error.flatten(), rcond=None)[0]

            # Update rotation and translation vectors
            delta_R = np.array([[1, -delta[2], delta[1]],
                                [delta[2], 1, -delta[0]],
                                [-delta[1], delta[0], 1]], dtype=np.float32)
            R[i] = np.dot(R[i], delta_R)
            t += delta[:3]

    return rvec.flatten(), t.flatten()

# Add your undistort_points and project_points functions here

def aruco_display(corners, ids, image):
    # Set coordinate system
    objPoints = np.zeros((4, 3), dtype=np.float32)
    markerLength = 5.0  # Example marker length, replace with actual value if available

    # Define coordinates for the four corners of the marker
    objPoints[:, :2] = np.array([[-markerLength/2, markerLength/2],
                                  [markerLength/2, markerLength/2],
                                  [markerLength/2, -markerLength/2],
                                  [-markerLength/2, -markerLength/2]], dtype=np.float32)

    #estimatePose=True
    nMarkers = len(corners)
    rvecs = [None] * nMarkers
    tvecs = [None] * nMarkers

    if ids is not None:
        for i, corner in enumerate(corners):
            #print(f"Marker ID: {ids[i]}, Number of Corners: {len(corner)}")
            if len(corner) >= 4:  # Check if enough corners are detected
                #print("Camera Matrix:", cameraMatrix)
                #print("Distortion Coefficients:", distCoeffs)
                
                #print("Length of objPoints array:", len(objPoints))
                #print("Length of corner array:", len(corner))
                #print("Contents of corner array:", corner)  # Add this line to print corner coordinates
                #print("The Object Points are as follows:",objPoints)
                if len(corner) < 4:
                    print("Not enough corners detected for marker {}. Skipping.".format(ids[i]))
                    continue

                rvecs[i], tvecs[i] = pnp_dlt(objPoints, np.array(corner).astype(np.float32), cameraMatrix, distCoeffs)
                #print("Length of rvecs:", len(rvecs[i]))
                #print("Length of tvecs:", len(tvecs[i]))
                #print("Length of rvecs:", (rvecs[i]))
                #print("Length of tvecs:", (tvecs[i]))

                rvec, tvec = rvecs[i], tvecs[i]
                markerLength_i = markerLength * 1.5
                
                # Project 3D points of axes onto the image plane
                axisPoints = np.float32([[0, 0, 0], [markerLength_i, 0, 0], [0, markerLength_i, 0], [0, 0, -markerLength_i]])
                axisPointsImg, _ = cv2.projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs)
                axisPointsImg = np.int32(axisPointsImg).reshape(-1, 2)
                
                # Draw lines representing the axes
                cv2.line(image, tuple(axisPointsImg[0]), tuple(axisPointsImg[1]), (0, 0, 255), 2)
                cv2.line(image, tuple(axisPointsImg[0]), tuple(axisPointsImg[2]), (255, 255, 60), 2)
                cv2.line(image, tuple(axisPointsImg[0]), tuple(axisPointsImg[3]), (255, 0, 0), 2)
                cv2.polylines(image, [np.array(corner)], True, (0, 255, 0), 2)
                cv2.putText(image, str(ids[i]), tuple(corner[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                #print("[Inference] ArUco marker ID: {}".format(ids[i]))
            else:
                print(f"Not enough corners detected for marker {ids[i]}")
                continue  # Skip further processing for this marker

    return image


def main():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            print("Error: Unable to read frame from camera")
            break

        h, w, _ = img.shape

        width = 1000
        height = int(width * (h / w))
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        corners, regions, ids = detect_markers(img)

        # Print the number of detected markers
        #print("Number of detected markers:", len(corners))

        # Skip further processing if no markers are detected
        if len(corners) == 0:
            continue

        # Iterate over detected markers and print the shape of each corner array
        for i, corner_array in enumerate(corners):
            print("Shape of corner array {}: {}".format(i, corner_array.shape))

        detected_markers = aruco_display(corners, ids, img)

        cv2.imshow("Image", detected_markers)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()
