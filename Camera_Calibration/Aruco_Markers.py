import cv2
import numpy as np
import pickle

with open('Camera_Calibration/cameraMatrix.pkl', 'rb') as f:
    cameraMatrix = pickle.load(f)
with open('Camera_Calibration/dist.pkl', 'rb') as f:
    distCoeffs = pickle.load(f)
with open('Camera_Calibration/calibration.pkl', 'rb') as f:
    calibration = pickle.load(f)
print("calibration: ",calibration)
    
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
                                   cv2.THRESH_BINARY, 7, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    corners, regions, ids = [], [], []
    for contour in contours:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4:  # Check if the contour has 4 vertices (i.e., a rectangle)
            x, y, w, h = cv2.boundingRect(approx)
            if w / h > 0.9 and w / h < 1.1:  # Aspect ratio close to 1 (i.e., approximately square)
                if cv2.contourArea(contour) > 1000 and cv2.contourArea(contour) < 4000:
                    corners.append(approx.reshape(-1, 2))
                    region = (x, y, w, h)
                    regions.append(region)
                    marker_region = thresh[y:y+h, x:x+w]

                    # Analyze marker bits
                    cells = divide_image_into_cells(marker_region, 7, 7)  # Assuming 7x7 marker size
                    bits = analyze_bits(cells)
                    print("Marker ID:", bits)
                    
                    # For demonstration purposes, just appending the sum of bits as an ID
                    ids.append(sum(bits))

    return corners, regions, ids

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
            print(f"Marker ID: {ids[i]}, Number of Corners: {len(corner)}")
            if len(corner) >= 4:  # Check if enough corners are detected
                print("Camera Matrix:", cameraMatrix)
                print("Distortion Coefficients:", distCoeffs)
                print("Length of objPoints array:", len(objPoints))
                print("Length of corner array:", len(corner))
                print("Contents of corner array:", corner)  # Add this line to print corner coordinates

                if len(corner) < 4:
                    print("Not enough corners detected for marker {}. Skipping.".format(ids[i]))
                    continue

                ret, rvecs[i], tvecs[i] = cv2.solvePnP(objPoints, np.array(corner).astype(np.float32), cameraMatrix, distCoeffs)
                print("Length of rvecs:", len(rvecs[i]))
                print("Length of tvecs:", len(tvecs[i]))
                print(" rvecs:", (rvecs[i]))
                print("tvecs:", (tvecs[i]))
                rvec, tvec = rvecs[i], tvecs[i]
                markerLength_i = markerLength * 1.5
                print("calibration: ",calibration)

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
                print("[Inference] ArUco marker ID: {}".format(ids[i]))
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
