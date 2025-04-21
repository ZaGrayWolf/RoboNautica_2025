import cv2  
import cv2.aruco as aruco
import numpy as np

# --- Hard-coded calibration parameters (from your XML file) ---

# Camera matrix (3x3)
cameraMatrix = np.array([
    [1199.8640834468974, 0, 319.5],
    [0, 1199.8640834468974, 239.5],
    [0, 0, 1]
], dtype=np.float32)

# Distortion coefficients (5x1)
distCoeffs = np.array([0.051011535316695424, 2.9740831958923337, 0, 0, -49.154706776004986], dtype=np.float32)

# --- ArUco marker detection and pose estimation functions ---

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
    if len(cell.shape) > 2:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    _, binary_cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if 0.9 < w / h < 1.1:
                if 1000 < cv2.contourArea(contour) < 4000:
                    corners.append(approx.reshape(-1, 2))
                    region = (x, y, w, h)
                    regions.append(region)
                    marker_region = thresh[y:y+h, x:x+w]
                    cells = divide_image_into_cells(marker_region, 7, 7)  # assuming 7x7 cells in marker
                    bits = analyze_bits(cells)
                    print("Marker bits (as ID candidate):", bits)
                    # For demonstration, use sum(bits) as a dummy ID
                    ids.append(sum(bits))
    return corners, regions, ids

def aruco_display(corners, ids, image):
    # Use a fixed marker length for pose estimation (example value)
    markerLength = 5.0  # This is in your chosen metric (e.g., centimeters or arbitrary units)
    
    # Define 3D object points for a square marker
    objPoints = np.array([[-markerLength/2, markerLength/2, 0],
                          [markerLength/2, markerLength/2, 0],
                          [markerLength/2, -markerLength/2, 0],
                          [-markerLength/2, -markerLength/2, 0]], dtype=np.float32)
    
    nMarkers = len(corners)
    rvecs = [None] * nMarkers
    tvecs = [None] * nMarkers

    if ids is not None:
        for i, corner in enumerate(corners):
            print(f"Marker ID: {ids[i]}, Corners count: {len(corner)}")
            if len(corner) >= 4:
                # Solve for pose using cv2.solvePnP
                ret, rvec, tvec = cv2.solvePnP(objPoints, np.array(corner).astype(np.float32), cameraMatrix, distCoeffs)
                if ret:
                    rvecs[i] = rvec
                    tvecs[i] = tvec
                    # For display, extend the marker length for axis visualization
                    markerLength_i = markerLength * 1.5
                    axisPoints = np.float32([[0, 0, 0],
                                               [markerLength_i, 0, 0],
                                               [0, markerLength_i, 0],
                                               [0, 0, -markerLength_i]])
                    axisPointsImg, _ = cv2.projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs)
                    axisPointsImg = np.int32(axisPointsImg).reshape(-1, 2)
                    # Draw axes
                    cv2.line(image, tuple(axisPointsImg[0]), tuple(axisPointsImg[1]), (0, 0, 255), 2)
                    cv2.line(image, tuple(axisPointsImg[0]), tuple(axisPointsImg[2]), (255, 255, 60), 2)
                    cv2.line(image, tuple(axisPointsImg[0]), tuple(axisPointsImg[3]), (255, 0, 0), 2)
                    cv2.polylines(image, [np.array(corner)], True, (0, 255, 0), 2)
                    cv2.putText(image, str(ids[i]), tuple(corner[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    print("[Inference] ArUco marker ID (dummy): {}".format(ids[i]))
                else:
                    print(f"Pose estimation failed for marker {ids[i]}")
            else:
                print(f"Not enough corners for marker {ids[i]}")
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
        print("Number of detected markers:", len(corners))

        if len(corners) == 0:
            cv2.imshow("Image", img)
        else:
            for i, corner_array in enumerate(corners):
                print("Shape of corner array {}: {}".format(i, corner_array.shape))
            detected_markers = aruco_display(corners, ids, img)
            cv2.imshow("Image", detected_markers)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
