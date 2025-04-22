import cv2
import numpy as np
import pickle

with open('Camera_Calibration/cameraMatrix.pkl', 'rb') as f:
    cameraMatrix = pickle.load(f)
with open('Camera_Calibration/dist.pkl', 'rb') as f:
    distCoeffs = pickle.load(f) 
    
print(cameraMatrix)
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

    # Extract camera parameters
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    distortion_coeffs = (3.48533793e+00, 8.70974618e+01, 2.37235072e-01, -1.36174455e-01, 9.91264335e+02)
    k1, k2, p1, p2, k3 = distortion_coeffs

    # Undistort each point
    undistorted_points = []
    for x, y in image_points_np:
        # Normalize image coordinates
        xn = (x - cx) / fx
        yn = (y - cy) / fy

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

    return np.array(undistorted_points)


def pnp_dlt(objpoints, imgPoints, cameraMatrix, distCoeff):
    
    inv_camera_matrix = np.linalg.inv(cameraMatrix)
    imgPoints = undistort_points(cameraMatrix, distCoeff, imgPoints)
    f=8.78062978e+3
    # Convert imgPoints to numpy array
    imgPoints_np = np.array(imgPoints)
    
    # Convert to grayscale if not already
    if len(imgPoints_np.shape) == 3:
        gray = cv2.cvtColor(imgPoints_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = imgPoints_np
    
    # Ensure gray is of type np.uint8
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)
    
    # Find contours
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Assuming only one contour is found
    contour = contours[0]
    
    # Calculate area of contour
    area = cv2.contourArea(contour)

    # Assuming z_values is a scalar value
    z_values_scalar = 25 * f / area

     # Convert scalar value to a 4x1 matrix
    z_values_matrix = np.full((4, 1), z_values_scalar)

    # Print the resulting matrix
    print(z_values_matrix)

    # Extracting x and y values
    x_values = objpoints[:, 0]  # Extracts the x values from all rows
    y_values = objpoints[:, 1]  # Extracts the y values from all rows

    # Creating the 4x1 array of [x y 0 1]
    points_4x1 = np.column_stack((x_values, y_values, z_values_matrix, np.ones_like(x_values))).T
    x_value = imgPoints[:, 0]  # Extracts the x values from all rows
    y_value = imgPoints[:, 1]  # Extracts the y values from all rows
    # Creating the 3x1 array of [x y 1]
    points_3x1 = np.column_stack((x_value, y_value, np.ones_like(x_values))).T

    print("4x1 array:")
    print(points_4x1)
    print("\n3x1 array:")
    print(points_3x1)
    for i in range(points_4x1.shape[1]):
        column = points_4x1[:, i]
        print("Column {}: {}".format(i+1, column))

# Accessing each column of the 3x1 array
    for i in range(points_3x1.shape[1]):
        column = points_3x1[:, i]
        print("Column {}: {}".format(i+1, column))
    # Initialize a list to store 3x12 matrices
    matrix_list = []

    # Loop over each column of the 3x1 matrix
    for i in range(points_3x1.shape[1]):
        # Extract the current column
        column = points_3x1[:, i]
        
        # Convert the column into a skew-symmetric matrix
        skew_symmetric_matrix = np.array([
            [0, -column[2], column[1]],
            [column[2], 0, -column[0]],
            [-column[1], column[0], 0]
        ])
        
        # Append the skew-symmetric matrix to the list
        matrix_list.append(skew_symmetric_matrix)

    #print("Skew-symmetric matrices:")
    #for i, matrix in enumerate(matrix_list):
        #print("Matrix {}:".format(i+1))
        #print(matrix)
    
    # Initialize list to store 3x12 matrices
    matrices = []

    # Loop over each column of the 4x1 matrix
    for i in range(points_4x1.shape[1]):
        # Extract the current column
        column = points_4x1[:, i]
        
        # Initialize list to store the rows of the current matrix
        rows = []

        # Fill the rows according to the specifications
        for j in range(3):  # We want 3x12 matrices
            if j == 0:
                row = list(column) + [0] * 8  # Fill the first row with the column values
            elif j == 1:
                row = [0] * 4 + list(column) + [0] * 4  # Fill the second row with the column values
            else:
                row = [0] * 8 + list(column)  # Fill the third row with the column values
            
            # Append the row to the list of rows
            rows.append(row)
        
        # Convert the list of rows into a numpy array
        matrix = np.array(rows)
        
        # Append the matrix to the list of matrices
        matrices.append(matrix)

    # Convert the list of matrices into a numpy array
    matrices_array = np.array(matrices)

    #print("Array of 3x12 matrices:")
    #print(matrices_array)

    # Initialize list to store the results
    results = []

    # Perform matrix multiplication for each 3x12 matrix
    for matrix in matrices_array:
        for skew_symmetric_matrix in matrix_list:
            result = np.dot(skew_symmetric_matrix, matrix)
            results.append(result)

    # Convert the list of results into a numpy array
    results_array = np.array(results)

    # Print the results
    #print("Results of matrix multiplication:")
    #for i, result in enumerate(results_array):
        #print(f"Matrix {i+1}:")
        #print(result)
     #   print()
    # Concatenate the stacked matrices vertically to form a 12x12 matrix
    repeated_matrices=[np.repeat(matrix, 3, axis=0) for matrix in results_array]
    stacked_matrix = np.vstack(repeated_matrices)

    # Print the resulting 12x12 matrix
    print("Stacked 12x12 matrix:")
    print(stacked_matrix)
    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(stacked_matrix)

    # Extract the last vertical column (last right singular vector)
    last_column = Vt[-1]

    # Print the last vertical column
    #print("Last vertical column:")
    #print(last_column)
    # Reshape the last column into a 3x4 matrix
    reshaped_matrix = last_column.reshape(3, 4)

    # Print the reshaped matrix
    #print("Reshaped last_column into a 3x4 matrix:")
    #print(reshaped_matrix)
    rotation_3x3 = reshaped_matrix[:, :3]  # First 3 columns
    matrix_3x1 = reshaped_matrix[:, 3:]  # Last column
    rotation_3x3 = np.dot(inv_camera_matrix,rotation_3x3)
    # Compute Singular Value Decomposition (SVD)
    U, D, Vt = np.linalg.svd(rotation_3x3)
    tvec=np.dot(inv_camera_matrix,matrix_3x1)
    Rotation = np.dot(U,Vt)
    print("Rotation Matrix",Rotation)
    #maxx = max_of_diagonal(matrix)
    rvec, _ = cv2.Rodrigues(Rotation)

    print("rvec",rvec)
    print("tvec",tvec)
    
    return rvec, tvec
    
    
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
                print("rvecs:", (rvecs[i]))
                print("tvecs:", (tvecs[i]))
                rvec, tvec = rvecs[i], tvecs[i]
                markerLength_i = markerLength * 1.5
                # Project 3D points of axes onto the image plane
                axisPoints = np.float32([[0, 0, 0], [markerLength_i, 0, 0], [0, markerLength_i, 0], [0, 0, -markerLength_i]])
                axisPointsImg, _ = cv2.projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs)
                axisPointsImg = np.int32(axisPointsImg).reshape(-1, 2)
                print("axisPoints",axisPointsImg)

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

        corners, _, ids = detect_markers(img)

        # Print the number of detected markers
        print("Number of detected markers:", len(corners))

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
