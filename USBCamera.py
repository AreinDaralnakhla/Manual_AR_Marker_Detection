import cv2
import datetime

class USBCamera:

    INPUT_CAMERA = 0
    INPUT_VIDEO  = 1
    OUTPUT_FILE  = 0
    OUTPUT_VIDEO = 1

    #
    # コンストラクタ
    #
    # @param deviceID : カメラ番号
    #
    def __init__(self, deviceID, width, height, use_api):
        self.deviceID      = deviceID
        self.inputMode     = self.INPUT_CAMERA
        self.outputMode    = self.OUTPUT_FILE
        self.use_api       = use_api
        self.vflip         = False
        self.hflip         = False
        self.save_original = True

        self.Open(width, height, None, use_api)
        
        # 出力ファイル用のヘッダー生成
        date = datetime.datetime.now()
        self.output_header = "%04d-%02d-%02d-%02d:%02d:%02d" % (date.year, date.month, date.day, date.hour, date.minute, date.second)
        
        # 出力ファイル用の連番
        self.image_count = 0

        # ビデオ出力フラグ(True: ビデオ出力中, False: ビデオ出力停止中)
        self.video_out = False

    def __del__(self):
        self.Close()

    #
    # カメラをクローズする関数
    #
    def Close(self):
        if self.capture.isOpened() is True:
            self.capture.release()

    #
    # カメラまたはビデオをオープンする関数
    #
    # @param width  : 画像の横サイズ
    # @param height : 画像の縦サイズ
    # @param name   : 動画ファイル名
    #
    def Open(self, width, height, name, use_api):
        if self.inputMode == self.INPUT_CAMERA:
            return self.OpenCamera(width, height, use_api)
        else:
            return self.OpenVideo (name, use_api)

    #
    # カメラをオープンする関数
    #
    # @param width  : 画像の横サイズ
    # @param height : 画像の縦サイズ
    #
    def OpenCamera(self, width, height, use_api):
        self.inoutMode = self.INPUT_CAMERA
        self.capture = cv2.VideoCapture(self.deviceID, use_api)

        if not self.capture:
            print('Camera open error')
            return False

        if self.capture.isOpened() is False:
            message = "Camera ID %d is not found." % self.deviceID
            print(message)
            return False

        self.image = self.capture.read()
        self.capture.set(cv2.CAP_PROP_FPS, 30)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        self.width    = width
        self.height   = height
        self.nchannels = 3

        return True

    #
    # ビデオをオープンする関数
    #
    # @param name : 動画ファイル名
    #
    def OpenVideo(self, name, use_api):
        self.inputMode = self.INPUT_VIDEO
        self.capture = cv2.VideoCapture(name, use_api)
        if self.capture.isOpened() is False:
            message = "Video %s is not found." % name
            print(message)
            return False

        self.width  = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

        return True

    #
    # カメラまたはビデオから画像を取得する関数
    #
    def CaptureImage(self):
        ret, self.image = self.capture.read()
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        if self.vflip is True and self.hflip is True:
            self.image = cv2.flip(self.image, -1)
        elif self.vflip is True:
            self.image = cv2.flip(self.image, 0)
        elif self.hflip is True:
            self.image = cv2.flip(self.image, 1)
        return ret, self.image

    # 
    # 動画出力の初期化を行う関数
    #
    def VideoOutStart(self):
        self.outputMode = self.OUTPUT_VIDEO
        self.out_video_name = self.output_header + "%05d.mp4" % self.image_count
        print(self.out_video_name)        
        self.image_count += 1
        codec = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.writer = cv2.VideoWriter(self.out_video_name, codec, 30, (int(self.width), int(self.height)))
        self.video_out = True

    #
    # 動画出力の終了処理を行う関数
    #
    def VideoOutEnd(self):
        self.video_out = False
        self.writer.release()

    #
    # 画像を出力する関数
    #
    def SaveImage(self, image):
        if self.outputMode == self.OUTPUT_VIDEO:
            self.writer.write(image)
        else:
            filename = self.output_header + "-%05d.png" % self.image_count
            self.image_count += 1
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if self.save_original:
                if self.vflip is True and self.hflip is True:
                    image = cv2.flip(image, -1)
                elif self.vflip is True:
                    image = cv2.flip(image, 0)
                elif self.hflip is True:
                    image = cv2.flip(image, 1)

            cv2.imwrite(filename, image)

    #
    # 画像のフリップ設定を行う関数
    # 
    def SetFlip (self, hflip, vflip):
        self.hflip = hflip
        self.vflip = vflip
