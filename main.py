import sys
import numpy as np
import cv2
from cv2 import aruco
import glfw
from mqoloader.loadmqo import LoadMQO
import Application

#
# アプリケーションで使用するパラメータ
#
image_width  = 640
image_height = 480
#use_api = cv2.CAP_DSHOW # Windowsで使用する場合こちらを使う
#use_api = 0             # Linuxで使用する場合はこちらを使う 
use_api = cv2.CAP_AVFOUNDATION

#
# アプリケーション設定
#
width  = image_width
height = image_height
app = Application.Application('Aruco marker AR', width, height, 0, use_api)

#
# 3次元モデルの設定
#
point_3D = np.array([(-52.5, 52.5, 0.0),
                     (-52.5, -52.5, 0.0),
                     (52.5, -52.5, 0.0),
                     (52.5, 52.5, 0.0)])
app.estimator.set_3D_points(point_3D)

model_filename = "./shiba.mqo"
model_scale = 0.1
app.use_normal = False
model = LoadMQO(model_filename, model_scale, app.use_normal)
app.set_mqo_model(model)

#
# アプリケーションのメインループ
#
while not app.glwindow.window_should_close():
    app.display_func(app.glwindow.window)    
    glfw.poll_events()

glfw.terminate()


