import cv2
import numpy as np
import pickle

POINT_VARIANCE_THRESHOLD = 1e-7
RANK_TOLERANCE = 1e-7
ORTHOGONALITY_SQUARED_ERROR_THRESHOLD = 1e-7
SQRT3 = np.sqrt(3)

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


    
def __init__(self):
        self.num_null_vectors_ = -1
        self.num_solutions_ = 0

def solve(self, objectPoints, imagePoints, rvecs, tvecs):
        # Input checking
        objType = objectPoints.getMat().dtype
        imgType = imagePoints.getMat().dtype

        assert objType == np.float32 or objType == np.float64, "Type of objectPoints must be CV_32FC3 or CV_64FC3"
        assert imgType == np.float32 or imgType == np.float64, "Type of imagePoints must be CV_32FC2 or CV_64FC2"
        assert objectPoints.rows() == 1 or objectPoints.cols() == 1, "ObjectPoints must be a row or column vector"
        assert objectPoints.rows() >= 3 or objectPoints.cols() >= 3, "ObjectPoints must have at least 3 elements"
        assert imagePoints.rows() == 1 or imagePoints.cols() == 1, "ImagePoints must be a row or column vector"
        assert imagePoints.rows() * imagePoints.cols() == objectPoints.rows() * objectPoints.cols(), "Number of objectPoints and imagePoints must match"

        _imagePoints = imagePoints.getMat().astype(np.float64) if imgType == np.float32 else imagePoints.getMat()
        _objectPoints = objectPoints.getMat().astype(np.float64) if objType == np.float32 else objectPoints.getMat()

        self.num_null_vectors_ = -1
        self.num_solutions_ = 0

        self.computeOmega(_objectPoints, _imagePoints)
        self.solveInternal(_objectPoints)

        depthRot = rvecs.depth() if rvecs.fixedType() else cv2.CV_64F
        depthTrans = tvecs.depth() if tvecs.fixedType() else cv2.CV_64F

        rvecs.create(self.num_solutions_, 1, cv2.CV_MAKETYPE(depthRot, 3 if rvecs.fixedType() and rvecs.kind() == cv2._InputArray.STD_VECTOR else 1))
        tvecs.create(self.num_solutions_, 1, cv2.CV_MAKETYPE(depthTrans, 3 if tvecs.fixedType() and tvecs.kind() == cv2._InputArray.STD_VECTOR else 1))

        for i in range(self.num_solutions_):
            rotation = np.reshape(np.array(self.solutions_[i].r_hat), (3, 3))
            rvec, _ = cv2.Rodrigues(rotation)
            rvecs.getMatRef(i)[:] = rvec
            tvecs.getMatRef(i)[:] = np.array(self.solutions_[i].t)
            
            
def nearestRotationMatrixSVD(e):
        e33 = np.reshape(e, (3, 3))
        _, u, vt = cv2.SVDecomp(e33, flags=cv2.SVD_FULL_UV)
        detuv = np.linalg.det(u) * np.linalg.det(vt)
        diag = np.eye(3)
        diag[2, 2] = detuv
        r33 = np.dot(np.dot(u, diag), vt)
        return r33.flatten()
            
def nearestRotationMatrixFOAM(e):
        adj_e = np.zeros((9,))
        det_e = (e[0] * e[4] * e[8] - e[0] * e[5] * e[7] - e[1] * e[3] * e[8] +
                e[2] * e[3] * e[7] + e[1] * e[6] * e[5] - e[2] * e[6] * e[4])

        if np.abs(det_e) < 1E-04:
            # Singular, handle with SVD
            return nearestRotationMatrixSVD(e)

        # Compute adjoint of e
        adj_e[0] = e[4] * e[8] - e[5] * e[7]
        adj_e[1] = e[2] * e[7] - e[1] * e[8]
        adj_e[2] = e[1] * e[5] - e[2] * e[4]
        adj_e[3] = e[5] * e[6] - e[3] * e[8]
        adj_e[4] = e[0] * e[8] - e[2] * e[6]
        adj_e[5] = e[2] * e[3] - e[0] * e[5]
        adj_e[6] = e[3] * e[7] - e[4] * e[6]
        adj_e[7] = e[1] * e[6] - e[0] * e[7]
        adj_e[8] = e[0] * e[4] - e[1] * e[3]

        # ||e||^2, ||adj(e)||^2
        e_sq = np.dot(e, e)
        adj_e_sq = np.dot(adj_e, adj_e)

        # Compute l_max with Newton-Raphson from FOAM's characteristic polynomial
        l = 0.5 * (e_sq + 3.0)
        if det_e < 0.0:
            l = -l

        for i in range(15):
            lprev = l
            tmp = (l * l - e_sq)
            p = (tmp * tmp - 8.0 * l * det_e - 4.0 * adj_e_sq)
            pp = 8.0 * (0.5 * tmp * l - det_e)

            l -= p / pp

            if np.abs(l - lprev) <= 1E-12 * np.abs(lprev):
                break

        # Compute rotation matrix R
        a = l * l + e_sq
        denom = l * (l * l - e_sq) - 2.0 * det_e
        denom = 1.0 / denom

        R = np.zeros((9,))
        R[0] = (a * e[0] + 2.0 * (l * adj_e[0] - e[0] * e[0])) * denom
        R[1] = (a * e[1] + 2.0 * (l * adj_e[1] - e[1] * e[1])) * denom
        R[2] = (a * e[2] + 2.0 * (l * adj_e[2] - e[2] * e[2])) * denom
        R[3] = (a * e[3] + 2.0 * (l * adj_e[3] - e[3] * e[3])) * denom
        R[4] = (a * e[4] + 2.0 * (l * adj_e[4] - e[4] * e[4])) * denom
        R[5] = (a * e[5] + 2.0 * (l * adj_e[5] - e[5] * e[5])) * denom
        R[6] = (a * e[6] + 2.0 * (l * adj_e[6] - e[6] * e[6])) * denom
        R[7] = (a * e[7] + 2.0 * (l * adj_e[7] - e[7] * e[7])) * denom
        R[8] = (a * e[8] + 2.0 * (l * adj_e[8] - e[8] * e[8])) * denom

        return R

def computeOmega(self, objectPoints, imagePoints):
    
        omega_ = np.zeros((9, 9), dtype=np.float64)
        qa_sum = np.zeros((3, 9), dtype=np.float64)

        sum_img = np.array([0, 0], dtype=np.float64)
        sum_obj = np.array([0, 0, 0], dtype=np.float64)
        sq_norm_sum = 0

        _imagePoints = imagePoints.getMat()
        _objectPoints = objectPoints.getMat()

        n = _objectPoints.shape[0] * _objectPoints.shape[1]

        for i in range(n):
            img_pt = _imagePoints[i]
            obj_pt = _objectPoints[i]

            sum_img += img_pt
            sum_obj += obj_pt

            x, y = img_pt
            X, Y, Z = obj_pt
            sq_norm = x * x + y * y
            sq_norm_sum += sq_norm

            X2 = X * X
            XY = X * Y
            XZ = X * Z
            Y2 = Y * Y
            YZ = Y * Z
            Z2 = Z * Z

            omega_[0, 0] += X2
            omega_[0, 1] += XY
            omega_[0, 2] += XZ
            omega_[1, 1] += Y2
            omega_[1, 2] += YZ
            omega_[2, 2] += Z2

            omega_[0, 6] += -x * X2
            omega_[0, 7] += -x * XY
            omega_[0, 8] += -x * XZ
            omega_[1, 7] += -x * Y2
            omega_[1, 8] += -x * YZ
            omega_[2, 8] += -x * Z2

            omega_[3, 6] += -y * X2
            omega_[3, 7] += -y * XY
            omega_[3, 8] += -y * XZ
            omega_[4, 7] += -y * Y2
            omega_[4, 8] += -y * YZ
            omega_[5, 8] += -y * Z2

            omega_[6, 6] += sq_norm * X2
            omega_[6, 7] += sq_norm * XY
            omega_[6, 8] += sq_norm * XZ
            omega_[7, 7] += sq_norm * Y2
            omega_[7, 8] += sq_norm * YZ
            omega_[8, 8] += sq_norm * Z2

            qa_sum[0, 0] += X
            qa_sum[0, 1] += Y
            qa_sum[0, 2] += Z

            qa_sum[0, 6] += -x * X
            qa_sum[0, 7] += -x * Y
            qa_sum[0, 8] += -x * Z
            qa_sum[1, 6] += -y * X
            qa_sum[1, 7] += -y * Y
            qa_sum[1, 8] += -y * Z

            qa_sum[2, 6] += sq_norm * X
            qa_sum[2, 7] += sq_norm * Y
            qa_sum[2, 8] += sq_norm * Z

        qa_sum[1, 3] = qa_sum[0, 0]
        qa_sum[1, 4] = qa_sum[0, 1]
        qa_sum[1, 5] = qa_sum[0, 2]
        qa_sum[2, 0] = qa_sum[0, 6]
        qa_sum[2, 1] = qa_sum[0, 7]
        qa_sum[2, 2] = qa_sum[0, 8]
        qa_sum[2, 3] = qa_sum[1, 6]
        qa_sum[2, 4] = qa_sum[1, 7]
        qa_sum[2, 5] = qa_sum[1, 8]

        omega_[1, 6] = omega_[0, 7]
        omega_[2, 6] = omega_[0, 8]
        omega_[2, 7] = omega_[1, 8]
        omega_[4, 6] = omega_[3, 7]
        omega_[5, 6] = omega_[3, 8]
        omega_[5, 7] = omega_[4, 8]
        omega_[7, 6] = omega_[6, 7]
        omega_[8, 6] = omega_[6, 8]
        omega_[8, 7] = omega_[7, 8]

        omega_[3, 3] = omega_[0, 0]
        omega_[3, 4] = omega_[0, 1]
        omega_[3, 5] = omega_[0, 2]
        omega_[4, 4] = omega_[1, 1]
        omega_[4, 5] = omega_[1, 2]
        omega_[5, 5] = omega_[2, 2]

        omega_ = omega_ + np.transpose(qa_sum) @ np.linalg.pinv(qa_sum)

        _, s_, u_ = np.linalg.svd(omega_, full_matrices=True)
        u_ = u_.T

        num_null_vectors_ = sum(s_[7 - num_null_vectors_] < RANK_TOLERANCE for num_null_vectors_ in range(7))
        assert num_null_vectors_ < 6

        point_mean_ = sum_obj / n

        return omega_, qa_sum, num_null_vectors_, point_mean_

def solveInternal(self, objectPoints):
        min_sq_err = float('inf')
        num_eigen_points = self.num_null_vectors_ if self.num_null_vectors_ > 0 else 1

        for i in range(9 - num_eigen_points, 9):
            e = SQRT3 * self.u_[:, i]
            orthogonality_sq_err = orthogonality_error(e)

            solutions = []

            if orthogonality_sq_err < ORTHOGONALITY_SQUARED_ERROR_THRESHOLD:
                solutions.append(SQPSolution(r_hat=det3x3(e) * e, t=self.p_ * (det3x3(e) * e)))
                self.checkSolution(solutions[0], objectPoints, min_sq_err)
            else:
                r = np.zeros((9, 1))
                nearestRotationMatrixFOAM(e, r)
                solutions.append(self.runSQP(r))
                solutions[0].t = self.p_ * solutions[0].r_hat
                self.checkSolution(solutions[0], objectPoints, min_sq_err)

                r = np.zeros((9, 1))
                nearestRotationMatrixFOAM(-e, r)
                solutions.append(self.runSQP(r))
                solutions[1].t = self.p_ * solutions[1].r_hat
                self.checkSolution(solutions[1], objectPoints, min_sq_err)

        index, c = 1, 1
        while index > 0 and min_sq_err > 3 * self.s_[index]:
            e = self.u_[:, index]
            solutions = []

            r = np.zeros((9, 1))
            nearestRotationMatrixFOAM(e, r)
            solutions.append(self.runSQP(r))
            solutions[0].t = self.p_ * solutions[0].r_hat
            self.checkSolution(solutions[0], objectPoints, min_sq_err)

            r = np.zeros((9, 1))
            nearestRotationMatrixFOAM(-e, r)
            solutions.append(self.runSQP(r))
            solutions[1].t = self.p_ * solutions[1].r_hat
            self.checkSolution(solutions[1], objectPoints, min_sq_err)

            index -= 1
            c += 1   
            
def det3x3(e):
        return e[0] * e[4] * e[8] + e[1] * e[5] * e[6] + e[2] * e[3] * e[7] \
            - e[6] * e[4] * e[2] - e[7] * e[5] * e[0] - e[8] * e[3] * e[1]
    
def orthogonality_error(e):
    sq_norm_e1 = e[0] * e[0] + e[1] * e[1] + e[2] * e[2]
    sq_norm_e2 = e[3] * e[3] + e[4] * e[4] + e[5] * e[5]
    sq_norm_e3 = e[6] * e[6] + e[7] * e[7] + e[8] * e[8]
    dot_e1e2 = e[0] * e[3] + e[1] * e[4] + e[2] * e[5]
    dot_e1e3 = e[0] * e[6] + e[1] * e[7] + e[2] * e[8]
    dot_e2e3 = e[3] * e[6] + e[4] * e[7] + e[5] * e[8]

    return (sq_norm_e1 - 1) ** 2 + (sq_norm_e2 - 1) ** 2 + (sq_norm_e3 - 1) ** 2 + \
           2 * (dot_e1e2 ** 2 + dot_e1e3 ** 2 + dot_e2e3 ** 2)
           
SQP_SQUARED_TOLERANCE = 1e-6
SQP_MAX_ITERATION = 100
SQP_DET_THRESHOLD = 1e-6

def run_SQP(r0):
    r = np.array(r0)

    delta_squared_norm = np.inf
    step = 0

    while delta_squared_norm > SQP_SQUARED_TOLERANCE and step < SQP_MAX_ITERATION:
        delta = solve_SQP_system(r)
        r += delta
        delta_squared_norm = np.linalg.norm(delta) ** 2
        step += 1

    solution = SQPSolution(np.zeros_like(r))

    det_r = det3x3(r)
    if det_r < 0:
        r = -r
        det_r = -det_r

    if det_r > SQP_DET_THRESHOLD:
        solution.r_hat = nearestRotationMatrixFOAM(r)
    else:
        solution.r_hat = r

    return solution

def solve_SQP_system(r, delta):
    sqnorm_r1 = np.dot(r[:3], r[:3])
    sqnorm_r2 = np.dot(r[3:6], r[3:6])
    sqnorm_r3 = np.dot(r[6:], r[6:])
    dot_r1r2 = np.dot(r[:3], r[3:6])
    dot_r1r3 = np.dot(r[:3], r[6:])
    dot_r2r3 = np.dot(r[3:6], r[6:])

    N = np.zeros((9, 3))
    H = np.zeros((9, 6))
    JH = np.zeros((6, 6))

    computeRowAndNullspace(r, H, N, JH)

    g = np.array([1 - sqnorm_r1, 1 - sqnorm_r2, 1 - sqnorm_r3, -dot_r1r2, -dot_r2r3, -dot_r1r3])

    x = np.zeros(6)
    x[0] = g[0] / JH[0, 0]
    x[1] = g[1] / JH[1, 1]
    x[2] = g[2] / JH[2, 2]
    x[3] = (g[3] - JH[3, 0] * x[0] - JH[3, 1] * x[1]) / JH[3, 3]
    x[4] = (g[4] - JH[4, 1] * x[1] - JH[4, 2] * x[2] - JH[4, 3] * x[3]) / JH[4, 4]
    x[5] = (g[5] - JH[5, 0] * x[0] - JH[5, 2] * x[2] - JH[5, 3] * x[3] - JH[5, 4] * x[4]) / JH[5, 5]

    delta[:] = np.dot(H, x)

    nt_omega = np.dot(N.T, omega_)
    W = np.dot(nt_omega, N)
    W_inv = np.zeros((3, 3))

    analytical_inverse_3x3_symm(W, W_inv)

    y = -np.dot(np.dot(W_inv, nt_omega), delta + r)
    delta += np.dot(N, y)

    return delta

def computeRowAndNullspace(r, norm_threshold=0.3):
    H = np.zeros((9, 6))
    N = np.zeros((9, 3))
    K = np.zeros((6, 6))

    # 1. q1
    norm_r1 = np.linalg.norm(r[:3])
    inv_norm_r1 = 1.0 / norm_r1 if norm_r1 > 1e-5 else 0.0
    H[:3, 0] = r[:3] * inv_norm_r1
    K[0, 0] = 2 * norm_r1

    # 2. q2
    norm_r2 = np.linalg.norm(r[3:6])
    inv_norm_r2 = 1.0 / norm_r2
    H[3:6, 1] = r[3:6] * inv_norm_r2
    K[1, 1] = 2 * norm_r2

    # 3. q3
    q3 = np.cross(H[3:6, 1], r[6:])
    norm_q3 = np.linalg.norm(q3)
    H[6:, 2] = q3 / norm_q3
    K[2, 2] = 2 * norm_q3

    # 4. q4
    q4 = r[3:6] - np.dot(r[3:6], H[:3, 0]) * H[:3, 0]
    q4_norm = np.linalg.norm(q4)
    H[:3, 3] = q4 / q4_norm
    K[3, :3] = np.dot(r[3:6], H[:3, 0])
    K[3, 3] = np.dot(r[3:6], H[:3, 3])

    # 5. q5
    q5 = r[6:] - np.dot(r[6:], H[3:6, 1]) * H[3:6, 1] - np.dot(r[6:], H[:3, 3]) * H[:3, 3]
    q5_norm = np.linalg.norm(q5)
    H[:3, 4] = q5 / q5_norm
    K[4, :3] = np.dot(r[6:], H[:3, 4])
    K[4, 3] = np.dot(r[6:], H[3:6, 1])
    K[4, 4] = np.dot(r[6:], H[3:6, 4]) + np.dot(r[6:], H[:3, 3])

    # 6. q6
    q6 = r[:3] - np.dot(r[:3], H[:3, 0]) * H[:3, 0] - np.dot(r[:3], H[3:6, 2]) * H[3:6, 2] - np.dot(r[:3], H[:3, 3]) * H[:3, 3]
    q6_norm = np.linalg.norm(q6)
    H[:3, 5] = q6 / q6_norm
    K[5, :3] = np.dot(r[:3], H[:3, 5])
    K[5, 3] = np.dot(r[:3], H[:3, 3])
    K[5, 4] = np.dot(r[:3], H[3:6, 1])
    K[5, 5] = np.dot(r[:3], H[3:6, 4]) + np.dot(r[:3], H[:3, 5])

    # Null space computation
    Pn = np.eye(9) - np.dot(H, H.T)
    col_norms = np.linalg.norm(Pn, axis=0)

    indices = np.argsort(col_norms)[::-1]
    selected_indices = []
    for idx in indices:
        if col_norms[idx] > norm_threshold:
            selected_indices.append(idx)
            if len(selected_indices) == 3:
                break

    v1 = Pn[:, selected_indices[0]]
    N[:, 0] = v1 / np.linalg.norm(v1)

    v2 = Pn[:, selected_indices[1]] - np.dot(N[:, 0], Pn[:, selected_indices[1]]) * N[:, 0]
    N[:, 1] = v2 / np.linalg.norm(v2)

    v3 = Pn[:, selected_indices[2]] - np.dot(N[:, 0], Pn[:, selected_indices[2]]) * N[:, 0] - np.dot(N[:, 1], Pn[:, selected_indices[2]]) * N[:, 1]
    N[:, 2] = v3 / np.linalg.norm(v3)

    return H, N

def analytical_inverse_3x3_symm(Q, threshold=1e-6):
    # Get the elements of the matrix
    a = Q[0, 0]
    b = Q[1, 0]
    d = Q[1, 1]
    c = Q[2, 0]
    e = Q[2, 1]
    f = Q[2, 2]

    # Determinant
    t2 = e * e
    t4 = a * d
    t7 = b * b
    t9 = b * c
    t12 = c * c
    det = -t4 * f + a * t2 + t7 * f - 2.0 * t9 * e + t12 * d

    if abs(det) < threshold:
        return False, None

    # Inverse
    t15 = 1.0 / det
    t20 = (-b * f + c * e) * t15
    t24 = (b * e - c * d) * t15
    t30 = (a * e - t9) * t15

    Qinv = np.zeros((3, 3))
    Qinv[0, 0] = (-d * f + t2) * t15
    Qinv[0, 1] = Qinv[1, 0] = -t20
    Qinv[0, 2] = Qinv[2, 0] = -t24
    Qinv[1, 1] = -(a * f - t12) * t15
    Qinv[1, 2] = Qinv[2, 1] = t30
    Qinv[2, 2] = -(t4 - t7) * t15

    return True, Qinv
    
    
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

                rvecs[i], tvecs[i] = solve(objPoints, np.array(corner).astype(np.float32), cameraMatrix, distCoeffs)
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
