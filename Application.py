import numpy as np
import datetime
import cv2
from cv2 import aruco

from OpenGL.GL import *
import glfw

import USBCamera as cam
import GLWindow
import PoseEstimation as ps

from scipy.spatial.transform import Rotation as R

#
# MRアプリケーションクラス
#
class Application:

    # ------------------------------------------------------------------------
    # コンストラクタ
    # ------------------------------------------------------------------------
    # @param width    : 画像の横サイズ
    # @param height   : 画像の縦サイズ
    # @param deviceID : カメラ番号
    #
    def __init__(self, title, width, height, deviceID, use_api):
        if deviceID != -1:
            self.use_camera = True
        else:
            self.use_camera = False

        # 画像の大きさ設定
        self.width   = width
        self.height  = height
        self.channel = 3
        self.detection_count = 0  # Initialize counter in __init__
        self.previous_R = np.eye(3)  # Initialize as identity matrix

        


        # カメラの設定
        if self.use_camera:
            self.camera = cam.USBCamera (deviceID, width, height, use_api)

        #
        # GLウィンドウの設定
        # 
        self.glwindow = GLWindow.GLWindow(title, width, height, self.display_func, self.keyboard_func)

        #
        # カメラの内部パラメータ
        #
        self.focus = 700.0
        self.u0    = width / 2.0
        self.v0    = height / 2.0

        #
        # OpenGLの表示パラメータ
        #
        scale = 0.01
        self.viewport_horizontal = self.u0 * scale
        self.viewport_vertical   = self.v0 * scale
        self.viewport_near       = self.focus * scale
        self.viewport_far        = self.viewport_near * 1.0e+6
        self.modelview           = (GLfloat * 16)()
        self.draw_axis           = True
        self.use_normal          = False

        #
        # カメラ姿勢を推定するクラス変数
        #
        self.estimator = ps.PoseEstimation(self.focus, self.u0, self.v0)
        
        # ファイル出力数のカウント用変数
        self.count = 0

        #
        # Arucoマーカーの設定
        #
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.aruco_parameters = aruco.DetectorParameters()
        self.ids = None
        self.corners = ()

    # ------------------------------------------------------------------------
    # カメラの内部パラメータの設定関数
    # ------------------------------------------------------------------------
    def SetCameraParam(self, focus, u0, v0):
        self.focus = focus
        self.u0    = u0
        self.v0    = v0

    # ------------------------------------------------------------------------
    # Manual Marker Detection
    # ------------------------------------------------------------------------

    def detect_marker_custom(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        marker_corners = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Skip small or excessively large areas
            if area < 1500:
                continue

            # Approximating contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Ensure the detected contour has 4 vertices (potential square)
            if len(approx) == 4:
                approx = self.sort_corners(approx)

                # Check if contour forms a square (aspect ratio)
                w = np.linalg.norm(approx[1] - approx[0])
                h = np.linalg.norm(approx[3] - approx[0])
                aspect_ratio = w / h

                if 0.9 < aspect_ratio < 1.1:  # Accept only near-square contours
                    # Homography warp (rectify tilted marker)
                    target_size = 400
                    destination_corners = np.array([
                        [0, 0],
                        [target_size - 1, 0],
                        [target_size - 1, target_size - 1],
                        [0, target_size - 1]
                    ], dtype='float32')

                    h, _ = cv2.findHomography(approx.reshape(4, 2), destination_corners)
                    warped = cv2.warpPerspective(frame, h, (target_size, target_size))

                    # Visualize for debugging
                    cv2.imshow("Warped Marker", warped)
                    cv2.waitKey(1)

                    # Solidity check (is contour filled and well-defined)
                    solidity = area / cv2.contourArea(cv2.convexHull(contour))
                    if solidity > 0.85:  # Ensure solid shape (not overly jagged)
                        marker_corners.append(approx)
                        break  # Stop after detecting the first marker

        return marker_corners


    def sort_corners(self, corners):
        # Sort corners: top-left, top-right, bottom-right, bottom-left
        corners = sorted(corners, key=lambda x: x[0][1])  # Sort by y (vertical)
        top_corners = sorted(corners[:2], key=lambda x: x[0][0])  # Top 2 by x (left to right)
        bottom_corners = sorted(corners[2:], key=lambda x: x[0][0])  # Bottom 2
        return np.array([top_corners[0], top_corners[1], bottom_corners[1], bottom_corners[0]])


    # ------------------------------------------------------------------------
    # Calibration Function
    """
    Manual Camera Calibration

    This function attempts to calibrate the camera using a known square marker
    of `marker_size` mm. We detect the marker corners in image space
    (img_points) and map them to the known world coordinates (obj_points).
    We then call `calibrate_camera_custom(...)` to compute a homography H,
    which is further decomposed into the camera's intrinsic matrix K and
    (R, t) for orientation and position.

    1) `objp` is the known 3D coordinates (or 2D if planar) of the square
       marker's corners in real-world units.
    2) `img_points` are the corresponding 2D points in the image plane.
    3) The function `ps.calibrate_camera_custom(...)` computes the homography
       from these correspondences, and then decomposes it into K, R, t.
    4) The resulting K (camera matrix) is stored in `self.estimator.A`, and
       we set `self.estimator.ready = True` to indicate successful calibration.
    """
    # Code that collects corners, calls ps.calibrate_camera_custom(), etc.

    # ------------------------------------------------------------------------

    def calibrate_camera(self):
        marker_size = 100
        frame_size = (self.width, self.height)
        
        obj_points = []
        img_points = []

        objp = np.array([
            [0, 0], 
            [marker_size, 0], 
            [marker_size, marker_size], 
            [0, marker_size]
        ], dtype=np.float32)

        if self.corners:
            for corner in self.corners:
                obj_points.append(objp)
                img_points.append(corner[:, 0])

            try:
                print("Calibrating with detected points...")
                K, R, t = ps.calibrate_camera_custom(obj_points, img_points)
                self.estimator.A = K
                self.estimator.ready = True
                print("Camera Matrix (Intrinsic):\n", K)
                print("Rotation:\n", R)
                print("Translation:\n", t)
                print("Calibration successful.")
            except Exception as e:
                print(f"Calibration failed: {e}")



    # ------------------------------------------------------------------------
    # カメラ映像を表示するための関数
    #
    # ここに作成するアプリケーションの大部分の処理を書く
    # ------------------------------------------------------------------------
    
    def display_func(self, window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Load images from camera/ OpenGL window
        success = False
        if self.use_camera:
            success, self.image = self.camera.CaptureImage()
        else:
            self.image = self.glwindow.image

        if not success:
            return

        # Call the manual-marker detection function 
        self.corners = self.detect_marker_custom(self.image)
        # print(f"Detected Contours: {len(self.corners)}")

        self.ids = [1] if self.corners else None


        # Auto-calibrate on marker detection
        if self.ids is not None:
            if not self.estimator.ready:
                print("Marker detected. Performing auto-calibration...")  # Check if this prints
                self.calibrate_camera()
            else:
                print("Calibration already done. Reprinting calibration data...")
                print(f"Camera Matrix:\n{self.estimator.A}")

        self.glwindow.draw_image(self.image)

        # Perform pose estimation and render model
        if self.ids is not None:
            success = self.compute_camera_pose()
            print(f"Camera pose computation success: {success}")

            if success:
                # Draw 3D model
                self.draw_3D_model()
        glfw.swap_buffers(window)

    # ------------------------------------------------------------------------
    # キー関数
    # ------------------------------------------------------------------------
    def keyboard_func(self, window, key, scancode, action, mods):
        # Qで終了
        if key == glfw.KEY_Q:
            glfw.set_window_should_close(self.glwindow.window, GL_TRUE)

        # Sで画像の保存
        if action == glfw.PRESS and key == glfw.KEY_S:
            self.save_image(self.count)
            self.count += 1

        # Tでランドマーク表示を切り替え
        if action == glfw.PRESS and key == glfw.KEY_T:
            self.set_draw_landmark(not self.hand_detection.draw_landmark)

    # ------------------------------------------------------------------------
    # 画像を保存する関数
    # ------------------------------------------------------------------------
    def save_image(self, count):
        filename = 'output_image-%05d.png' % count
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        glReadBuffer (GL_BACK)
        glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE, image.data)
        image = cv2.flip (image, 0)
        image = cv2.cvtColor (image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, image)

    # ------------------------------------------------------------------------
    # mediapipeで検出した手のランドマークを描画するかを設定する関数
    # ------------------------------------------------------------------------
    def set_draw_landmark(self, draw_flag):
        self.hand_detection.set_draw_landmarks (draw_flag)

    # ------------------------------------------------------------------------
    # 3次元モデルをセットする関数
    # ------------------------------------------------------------------------
    def set_mqo_model(self, model):
        self.model = model

    # ------------------------------------------------------------------------
    # カメラ姿勢を推定する関数
    # ------------------------------------------------------------------------
    def compute_camera_pose(self):
        c = self.corners[0][:, 0]  
        x1, x2, x3, x4 = c[:,0]
        y1, y2, y3, y4 = c[:,1]
        
        point_2D = np.array([(x1, y1), (x2, y2), (x3, y3), (x4, y4)],
                            dtype = "double")

        # カメラ姿勢を計算
        success, R, t = self.estimator.compute_camera_pose(point_2D)

        if success:
            # 世界座標系に対するカメラ位置を計算
            #     この位置を照明位置として使用
            if self.use_normal:
                pos = -R.transpose().dot(t)
                self.camera_pos = np.array([pos[0], pos[1], pos[2], 1.0], dtype="double")
            # OpenGLで使用するモデルビュー行列を生成
            self.modelview[0] = R[0][0]
            self.modelview[1] = R[1][0]
            self.modelview[2] = R[2][0]
            self.modelview[3] = 0.0
            self.modelview[4] = R[0][1]
            self.modelview[5] = R[1][1]
            self.modelview[6] = R[2][1]
            self.modelview[7] = 0.0
            self.modelview[8] = R[0][2]
            self.modelview[9] = R[1][2]
            self.modelview[10] = R[2][2]
            self.modelview[11] = 0.0
            self.modelview[12] = t[0]
            self.modelview[13] = t[1]
            self.modelview[14] = t[2]
            self.modelview[15] = 1.0

        return success

    # ------------------------------------------------------------------------
    # カメラ姿勢を推定する関数
    # ------------------------------------------------------------------------
    def draw_3D_model(self):
        self.glwindow.push_GL_setting()
        
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()            
        glFrustum(-self.viewport_horizontal, self.viewport_horizontal, -self.viewport_vertical, self.viewport_vertical, self.viewport_near, self.viewport_far)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glLoadMatrixf(self.modelview)

        # 証明をオン
        if self.use_normal:
            glLightfv(GL_LIGHT0, GL_POSITION, self.camera_pos)
            glEnable(GL_LIGHTING)
            glEnable(GL_LIGHT0)

        model_shift_X = self.model.scale
        model_shift_Y = self.model.scale
        model_shift_Z = 0.0
                
        glTranslatef (model_shift_X, model_shift_Y, model_shift_Z)
        glRotatef (-90.0, 1.0, 0.0, 0.0)
        glScalef(self.model.scale, self.model.scale, self.model.scale)
        self.model.draw()

        self.glwindow.pop_GL_setting()
                
        # 証明をオフ
        if self.use_normal:                
            glDisable(GL_LIGHTING)
            glDisable(GL_LIGHT0)
