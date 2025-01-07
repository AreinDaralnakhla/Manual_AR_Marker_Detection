import numpy as np
import cv2

# 
# 平面上の特徴点対応からカメラ姿勢を推定するクラス
#
class PoseEstimation:

    # ------------------------------------------------------------------------
    # コンストラクタ
    # ------------------------------------------------------------------------
    def __init__(self, f, u0, v0):

        # 投影行列
        self.A = np.array([[f, 0.0, u0], [0.0, f, v0], [0.0, 0.0, 1.0]], dtype = "double")

        # 歪み係数
        self.dist_coeff = np.zeros((4, 1))

        # 3次元点と2次元点データ
        self.point_3D = np.array([])
        self.point_2D = np.array([])

        # 推定可能かどうかを表すフラグ
        self.ready = False

    # ------------------------------------------------------------------------
    # Custom calibration function
    # ------------------------------------------------------------------------
    def calibrate_camera_custom(obj_points, img_points):
        A = []  # Matrix for homography equations
        
        for i in range(len(obj_points)):
            X = obj_points[i]
            x = img_points[i]
            
            # Construct 2 rows for each point correspondence
            A.append([
                X[0], X[1], 1, 0, 0, 0, -x[0] * X[0], -x[0] * X[1], -x[0]
            ])
            A.append([
                0, 0, 0, X[0], X[1], 1, -x[1] * X[0], -x[1] * X[1], -x[1]
            ])
        
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1].reshape(3, 3)  # Last column of V (smallest singular value)
        
        # Decompose homography to extract camera parameters
        K, R, t = decompose_homography(h)
        
        print("Camera Matrix (Intrinsic):\n", K)
        print("Rotation:\n", R)
        print("Translation:\n", t)
        
        return K, R, t

    def decompose_homography(H):
        # Normalize H to ensure the last value is 1
        H /= H[-1, -1]

        # Extract column vectors
        h1 = H[:, 0]
        h2 = H[:, 1]
        h3 = np.cross(h1, h2)  # h3 is the cross product of h1 and h2
        lambda_ = 1 / np.linalg.norm(h1)
        
        # Construct intrinsic matrix K
        K = np.array([
            [np.linalg.norm(h1), 0, 0],
            [0, np.linalg.norm(h2), 0],
            [0, 0, 1]
        ])
        
        # Construct rotation matrix R
        R = np.array([h1, h2, h3]).T * lambda_
        
        # Compute translation
        t = H[:, 2] * lambda_
        
        return K, R, t


    # ------------------------------------------------------------------------
    # カメラ姿勢の推定関数
    # ------------------------------------------------------------------------
    def compute_camera_pose(self, point_2D):
        if self.ready:
            self.point_2D = point_2D
            success, vec_R, t = cv2.solvePnP(self.point_3D,
                                             self.point_2D,
                                             self.A,
                                             self.dist_coeff,
                                             flags = 0)
            R = cv2.Rodrigues (vec_R)[0]

            # OpenGLの座標系に変換する回転行列
            R_ = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
            R = np.dot(R_, R)
            t = np.dot(R_, t)
            
            return True, R, t
        else:
            return False, None, None

    # ------------------------------------------------------------------------
    # 3次元点をセットする関数
    # ------------------------------------------------------------------------
    def set_3D_points(self, point_3D):
        self.point_3D = point_3D
        self.ready = True
