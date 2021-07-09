import sys, os
import math
import numpy as np
import cv2
from numpy.core.fromnumeric import nonzero
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pyglet
import pyglet.gl as gl
import pyrealsense2 as rs

import ctypes
from PIL import Image
from plotter import Plotter


#===============================
# 定数
#===============================

TARGET_SCREEN_ID = 0     # プロジェクタのスクリーンID
CAMERA_ID = 1           # カメラID

DATA_DIRNAME = "data"
DATA_DIRPATH = os.path.join(os.path.dirname(__file__), DATA_DIRNAME)
if not os.path.exists(DATA_DIRPATH):
    os.makedirs(DATA_DIRPATH)

CHESS_HNUM = 7       # 水平方向個数
CHESS_VNUM = 10      # 垂直方向個数
CHESS_MARGIN = 50    # [px]h
CHESS_BLOCKSIZE = 80 # [px]

BOARD_WIDTH  = 0.33  # chessboard の横幅 [m]
BOARD_HEIGHT = 0.45  # chessboard の縦幅 [m]
BOARD_X = 0.         # chessboard の3次元位置X座標 [m]（右手系）
BOARD_Y = 0.         # chessboard の3次元位置Y座標 [m]（右手系）
BOARD_Z = -1.5       # chessboard の3次元位置Z座標 [m]（右手系）[see]

DEPTH_LIMIT = 2.0    # 点群処理をする最大の距離 [m]
MAX_PLANE_NUM = 3    # シーンから検出する平面の最大数

LARGE_VALUE = 99999      # 最適化の初期値（再投影誤差）
error_min = LARGE_VALUE  # 最適化の最良値（再投影誤差）




# OpenGL の射影のパラメータ
class Params:
    def __init__(self, zNear = 0.0001, zFar = 20.0, fovy = 21.0):
        self.Z_NEAR = zNear     # 最も近い点 [m]
        self.Z_FAR  = zFar      # 最も遠い点 [m]
        self.FOVY   = fovy      # 縦の視野角 [deg]


PARAMS = Params(zNear = 0.0001, # [m]
                zFar = 20.0,    # [m]
                fovy = 21.0     # [deg]
                )


#===============================
# クラス
#===============================
# realsense device
class Device:
    def __init__(self, pipeline, pipeline_profile):
        self.pipeline = pipeline
        self.pipeline_profile = pipeline_profile



#===============================
# 状態変数
#===============================
class AppState:

    OPT_SIMPLEX = 0x0001
    OPT_LM = 0x0002
    OPT_BASINHOPPING = 0x0004
    OPT_BRUTE = 0x0008

    def __init__(self, params):
        self.params = params
        self.zNear = self.params.Z_NEAR
        self.delta_zNear = 0.001

        self.roll = math.radians(0)
        self.pitch = math.radians(0)
        self.yaw = math.radians(0)
        self.tvec = np.array([0, 0, 0], np.float32)
        self.rvec = np.array([0, 0, 0], np.float32)

        # キャリブレーションパラメータ
        self.camera_frame = None
        self._camera_frame = None
        self.cp3d_opengl = None
        self.cp2d_projected = None
        self.cp2d_cpoint = None
        self.H_pj_cm = None

        # 描画時の状態変数
        self.mouse_btns = [False, False, False]
        self.draw_axes = False
        self.draw_grid = False
        self.draw_board = True

        self.half_fov = True                  # プロジェクタの画角の変数
        self.obtain_homography_matrix = False # ホモグラフィの推定モード
        self.set_control_points = False       # 標定点の設定モード

        self.opt_func = AppState.OPT_SIMPLEX

    def reset(self):
        self.zNear = self.params.Z_NEAR
        self.roll, self.pitch, self.yaw = 0, 0, 0
        self.tvec[:] = 0, 0, 0
        self.rvec[:] = np.array([0, 0, 0], np.float32)

    # @property
    # def rotation(self):
    #     Rx = rotation_matrix((1, 0, 0), math.radians(-self.pitch))
    #     Ry = rotation_matrix((0, 1, 0), math.radians(-self.yaw))
    #     return np.dot(Ry, Rx).astype(np.float32)



#===============================
# グローバル変数
#===============================
window = None        # pyglet の Window　クラスのインスタンス
state = None         # アプリの状態を管理する変数（AppState）
cam_w, cam_h  = 0, 0 # 画面解像度

# ボードのテクスチャ
board_texture = None
chessboard_data = None

projection_matrix = None
modelview_matrix = None

# [see] ボードの位置
board_vertices = ((BOARD_X - BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT, BOARD_Z),
                (BOARD_X - BOARD_WIDTH / 2, BOARD_Y, BOARD_Z),
                (BOARD_X + BOARD_WIDTH / 2, BOARD_Y, BOARD_Z),
                (BOARD_X + BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT, BOARD_Z))

camera = None


#-------------------------------
# Realsense
#-------------------------------
# RGB情報と深度情報の保存
def save_image(frame):
    color_path = os.path.join(DATA_DIRPATH, "color.png")
    depth_path = os.path.join(DATA_DIRPATH, "depth.png")

    data = frame.get_data()

    format = frame.get_profile().format()
    if format == rs.format.rgb8:
        color_image = np.asanyarray(data)
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(color_path, color_image)
        print('write rgb8')
    elif format == rs.format.bgr8:
        color_image = np.asanyarray(data)
        cv2.imwrite(color_path, color_image)
        print('write bgr8')
    elif format == rs.format.z16:
        depth_image = np.asanyarray(data)
        cv2.imwrite(depth_path, depth_image)
        print('write z16')
    return 1

# # Realsense の内部パラメータの確認と保存
# def show_and_save_intrinsic(intrinsic, saveFlag = False):
#     # 内部パラメータの表示
#     print("====================================")
#     print("intrinsic parameters")
#     print("====================================")
#     print("\tintrinsic.width", intrinsic.width)
#     print("\tintrinsic.height", intrinsic.height)
#     print("\tintrinsic.fx", intrinsic.fx)
#     print("\tintrinsic.fy", intrinsic.fy)
#     print("\tintrinsic.ppx", intrinsic.ppx)
#     print("\tintrinsic.ppy", intrinsic.ppy)

#     if saveFlag:
#         np.savez('intrinsic.npz', width = intrinsic.width, height = intrinsic.height, fx = intrinsic.fx, fy = intrinsic.fy, ppx = intrinsic.ppx, ppy = intrinsic.ppy)


def convert_fmt(fmt):
    """rs.format to pyglet format string"""
    return {
        rs.format.rgb8: 'RGB',
        rs.format.bgr8: 'BGR',
        rs.format.rgba8: 'RGBA',
        rs.format.bgra8: 'BGRA',
        rs.format.y8: 'L',
    }[fmt]


def run_realsense(dt):
    global cam_w, cam_h
    global image_data
    global depth_frame, color_frame
    global pcd
    
    window.set_caption("RealSense (%dx%d) %dFPS (%.2fms) %s" %
                       (cam_w, cam_h, 0 if dt == 0 else 1.0 / dt, dt * 1000,
                        "PAUSED" if state.paused else ""))

    # ポーズ
    if state.paused:
        return

    success, frames = pipeline.try_wait_for_frames(timeout_ms=0)
    if not success:
        return

    # フレームの取得
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    #color_frame = aligned_frames.first(color_stream)
    color_frame = aligned_frames.get_color_frame()

    #----------------------------------
    # フィルタ処理
    #----------------------------------
    # サブサンプリング
    depth_frame = decimate.process(depth_frame)

    # 後処理フィルタ
    if state.postprocessing:
        for f in filters:
            depth_frame = f.process(depth_frame)
    #----------------------------------

    # 内部パラメータの取得
    # Grab new intrinsics (may be changed by decimation)
    depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
    cam_w, cam_h = depth_intrinsics.width, depth_intrinsics.height

    # 画像の取得
    color_image = np.asanyarray(color_frame.get_data())
    depth_image = np.asanyarray(depth_frame.get_data())

    # 深度画像の取得
    colorized_depth = colorizer.colorize(depth_frame)
    depth_colormap = np.asanyarray(colorized_depth.get_data())

    if state.color:
        mapped_frame, color_source = color_frame, color_image
    else:
        mapped_frame, color_source = colorized_depth, depth_colormap

    # # 点群化
    # points = pc.calculate(depth_frame)
    # pc.map_to(mapped_frame)

    # verts = np.asarray(points.get_vertices(2)).reshape(cam_h, cam_w, 3)
    # texcoords = np.asarray(points.get_texture_coordinates(2))



# def run_realsense():
#     # フレーム待ち（color&depth）
#     # フレーム取得
#     frames = pipeline.wait_for_frames()
#     # フレームの画角差を修正
#     aligned_frames = align.process(frames)
#     # フレームの切り分け
#     # 多分これに射影変換行列をかけたら視点の変更ができる
#     color_frame = aligned_frames.get_color_frame()
#     depth_frame = aligned_frames.get_depth_frame()
#     # if not depth_frame or not color_frame:
#     #     continue

#     # RGB画像のフレームから画素値をnumpy配列に変換
#     # これで普通のRGB画像になる
#     color_image = np.asanyarray(color_frame.get_data())
#     # D画像のフレームから画素値をnumpy配列に変換
#     depth_image = np.asanyarray(depth_frame.get_data()) # 深度の画素値が入っている



#-------------------------------
# ここから描画関数
#-------------------------------

def on_draw_impl():
    window.clear()

    gl.glClearColor(0, 0, 0, 1)

    gl.glEnable(gl.GL_DEPTH_TEST)
    gl.glEnable(gl.GL_LINE_SMOOTH)

    projection()
    modelview()

    #====================================================
    if state.draw_board:
        board()

    # カメラ座標軸の描画
    if state.draw_axes and any(state.mouse_btns):
        axes(0.1, 4)

    # 地面の格子の描画
    if state.draw_grid:
        gl.glColor3f(0.5, 0.5, 0.5)
        grid()

    if state.draw_axes:
        gl.glColor3f(0.25, 0.25, 0.25)
        axes()
    #====================================================

    # 標定点の設定（at Pressed Key.C）
    if state.set_control_points:
        obtain_control_point()
        state.set_control_points = False

    # ホモグラフィ行列の推定（at Pressed Key.H）
    if state.obtain_homography_matrix:
        print("### Homography Matrix Estimation ###")
        obtain_homography_matrix()
        state.obtain_homography_matrix = False


#-------------------------------
# ここからがメイン部分
#-------------------------------
# メインの処理
if __name__ == '__main__':

    # ------------------------
    # RealSense
    # ------------------------
    WIDTH = 640 #RealSenseの縦横
    HEIGHT = 480

    # # color format
    # # データ形式の話
    # color_stream, color_format = rs.stream.color, rs.format.bgr8
    # depth_stream, depth_format = rs.stream.depth, rs.format.z16

    # # ストリーミング初期化
    # # RealSenseからデータを受信するための準備
    # # config.enable_streamでRGB，Dの解像度とデータ形式，フレームレートを指定している
    # config = rs.config()
    # config.enable_stream(depth_stream, WIDTH, HEIGHT, depth_format, 30)
    # config.enable_stream(color_stream, WIDTH, HEIGHT, color_format, 30)
    #
    # # ストリーミング開始
    # pipeline = rs.pipeline()
    # profile = pipeline.start(config)
    #
    # # 距離[m] = depth * depth_scale
    # depth_sensor = profile.get_device().first_depth_sensor()
    # depth_scale = depth_sensor.get_depth_scale()
    #
    # # Alignオブジェクト生成
    # # RGBとDの画角の違いによるズレを修正している
    # align_to = rs.stream.color
    # align = rs.align(align_to)
    # # max_dist = THRESHOLD / depth_scale
    #

    # アプリクラスのインスタンス
    state = AppState(PARAMS)
    #-------------------------------
    # ここから描画準備：Pyglet
    #-------------------------------
    display = pyglet.canvas.get_display()
    screens = display.get_screens()
    target_screen_id = TARGET_SCREEN_ID
    target_screen = screens[target_screen_id]  # ここで投影対象画面を変更
    config = gl.Config(
        double_buffer=True,
        sample_buffers=1,
        samples=4,  # MSAA
        depth_size=24,
        alpha_size=8
    )
    config = target_screen.get_best_config(config)
    window = pyglet.window.Window(
        config=config,
        resizable=True,
        vsync=False,
        fullscreen=True,
        screen=target_screen)

    @window.event
    def on_draw():
        on_draw_impl()