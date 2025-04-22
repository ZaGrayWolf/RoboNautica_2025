import numpy as np  
  
def normalize(points):  
    """ Normalize homogeneous coordinates (scale to make last element 1). """  
    return points / points[-1]  
  
def dlt(points_2d, points_3d):  

    if len(points_2d) < 6 or np.linalg.matrix_rank(points_3d) < 3:  
        raise ValueError("Insufficient or degenerate points")  
        
    """ Direct Linear Transform (DLT) algorithm """  
    assert len(points_2d) >= 6  
    assert len(points_3d) == len(points_2d)  
  
    # Check for degenerate configuration  
    if np.linalg.matrix_rank(points_3d) < 3:  
        raise ValueError("Points are in a degenerate configuration")  
  
    # Augment points_2d to homogeneous form  
    points_2d = np.hstack((points_2d, np.ones((points_2d.shape[0], 1))))  
  
    # Create matrix A  
    A = []  
    for (x, y, z), (u, v, _) in zip(points_3d, points_2d):  
        A.append([-x, -y, -z, -1, 0, 0, 0, 0, u*x, u*y, u*z, u])  
        A.append([0, 0, 0, 0, -x, -y, -z, -1, v*x, v*y, v*z, v])  
    A = np.array(A)  
  
    # Solve for projection matrix P (last eigenvector of A^T A)  
    _, _, V = np.linalg.svd(A)  
    P = V[-1].reshape(3, 4)  
  
    return P  
  
def project(points_3d, P):  
    """ Project 3D points to 2D using projection matrix P """  
    points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))  
    points_2d = np.dot(P, points_3d.T).T  
    points_2d = normalize(points_2d.T).T  
    return points_2d[:, :2]  
