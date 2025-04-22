import cv2
import numpy as np
import pickle
with open('Camera_Calibration/cameraMatrix.pkl', 'rb') as f:
    cameraMatrix = pickle.load(f)
with open('Camera_Calibration/dist.pkl', 'rb') as f:
    distCoeffs = pickle.load(f) 

def divide_image_into_cells(image, num_rows, num_cols):
    height, width = image.shape[:2]
    cell_height, cell_width = height // num_rows, width // num_cols
    return image.reshape(num_rows, cell_height, -1, cell_width).transpose(0, 2, 1, 3).reshape(-1, cell_height, cell_width)

def count_black_white_pixels(cell):
    if len(cell.shape) == 2:
        # Grayscale image, no conversion needed
        _, binary_cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        # Convert to grayscale if necessary
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
        _, binary_cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    num_black_pixels = np.sum(binary_cell == 0)
    num_white_pixels = np.sum(binary_cell == 255)
    return num_black_pixels, num_white_pixels

def analyze_bits(cells):
    bits = [count_black_white_pixels(cell)[0] > count_black_white_pixels(cell)[1] for cell in cells]
    return bits


def detect_markers(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Pre-allocate memory (adjust based on expected number of markers)
    estimated_markers = 5
    corners = [None] * estimated_markers
    regions = [None] * estimated_markers
    ids = []
    num_found = 0

    for i, contour in enumerate(contours):
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Combine operations (if applicable)
        if len(approx) == 4 and 0.9 < cv2.contourArea(contour) / cv2.contourArea(approx) < 1.1:
            x, y, w, h = cv2.boundingRect(approx)
            if 1500 < cv2.contourArea(contour) < 2500:
                corners[num_found] = approx.reshape(-1, 2)
                regions[num_found] = (x, y, w, h)
                marker_region = thresh[y:y+h, x:x+w]

                # Analyze marker bits (replace with your actual logic)
                cells = divide_image_into_cells(marker_region, 7, 7)  # Assuming 7x7 marker size
                bits = analyze_bits(cells)
                # print("Marker ID:", bits)

                # Calculate ID from bits (replace with your logic)
                ids.append(sum(bits))
                num_found += 1

                # Early termination (adjust based on needs)
                if num_found >= estimated_markers:
                    break

    return corners[:num_found], regions[:num_found], ids

def aruco_display(corners, ids, image, cameraMatrix, distCoeffs, markerLength=5.0):
    """
    Displays detected ArUco markers on the image with axes and IDs.

    Args:
        corners: List of detected marker corners (Nx4x2 format, where N is the number of markers).
        ids: List of corresponding marker IDs (length N).
        image: Input image.
        cameraMatrix: Camera calibration matrix.
        distCoeffs: Distortion coefficients.
        markerLength: Length of the marker in the real world (optional, default 5.0).

    Returns:
        The image with ArUco markers displayed.
    """

    nMarkers = len(corners)
    objPoints = np.zeros((4, 3), dtype=np.float32)
    objPoints[:, :2] = np.array([[-markerLength/2, markerLength/2],
                                 [markerLength/2, markerLength/2],
                                 [markerLength/2, -markerLength/2],
                                 [-markerLength/2, -markerLength/2]], dtype=np.float32)

    rvecs = [None] * nMarkers
    tvecs = [None] * nMarkers
    

    if ids is not None:
        for i, corner in enumerate(corners):
            if len(corner) >= 4:
                ret, rvec, tvec = cv2.solvePnP(objPoints, np.array(corner).astype(np.float32), cameraMatrix, distCoeffs)

                markerLength_i = markerLength * 1.5  # Optional scaling for axis visualization

                axisPoints = np.float32([[0, 0, 0], [markerLength_i, 0, 0], [0, markerLength_i, 0], [0, 0, -markerLength_i]])
                axisPointsImg, _ = cv2.projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs)
                axisPointsImg = np.int32(axisPointsImg).reshape(-1, 2)

                # Draw lines representing the axes and marker outline
                cv2.line(image, tuple(axisPointsImg[0]), tuple(axisPointsImg[1]), (0, 0, 255), 2)
                cv2.line(image, tuple(axisPointsImg[0]), tuple(axisPointsImg[2]), (255, 255, 60), 2)
                cv2.line(image, tuple(axisPointsImg[0]), tuple(axisPointsImg[3]), (255, 0, 0), 2)
                cv2.polylines(image, [np.array(corner)], True, (0, 255, 0), 2)

                # Draw marker ID
                cv2.putText(image, str(ids[i]), tuple(corner[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                print("[Inference] ArUco marker ID: {}".format(ids[i]))
            else:
                print(f"Not enough corners detected for marker {ids[i]}")

    return image


def main():
    cap = cv2.VideoCapture(0)

    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    # Pre-calculate aspect ratio
    ret, img = cap.read()
    if not ret:
        print("Error: Unable to read frame from camera")
        return
    h, w, _ = img.shape
    aspect_ratio = h / w

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            print("Error: Unable to read frame from camera")
            break

        # Resize efficiently
        width = 1000
        height = int(width * aspect_ratio)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)

        # Process markers
        corners, regions, ids = detect_markers(img)

        # Early exit if no markers detected
        if len(corners) == 0:
            continue

        # Display markers (separate function)
        detected_markers = aruco_display(corners, ids, img.copy())  # Use a copy to avoid modifying original image

        cv2.imshow("Image", detected_markers)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    cap.release()
