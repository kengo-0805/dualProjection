# 中間発表用にパラメータをいじったもの
#
"""
OpenGL Pointcloud viewer with http://pyglet.org

Usage:
------
Mouse:
    Drag with left button to rotate around pivot (thick small axes),
    with right button to translate and the wheel to zoom.

Keyboard:
    [f1]    Toggle full screen
    [f2]    Switch full screen
    [a]     Toggle axis, frustrum, grid
    [b]     Toggle chessbord
    [c]     2. Set control plots on screen captured by web-cam (manual plot)
    [g]     Toggle drawing grid floor
    [h]     1. Estimation of homography matrix H_pj-cm  (manual plot)
    [i]     Show current model coordinate position
    [j]     Visual check control points on web-cam image
    [k]     Visual check optimization result
    [o]     Start optimization
    [p]     Visual check control points on CG space
    [r]     Reset View
    [s]     Save PNG (./out.png)
    [x]     Load Optimization results (./data/parameters.npz)
    [z]     Save Optimization results (./data/parameters.npz)
    [→]     increase delta_zNear to zoom
    [←]     dencrease delta_zNear to zoom
    [↑]     unzoom by increasing zNear (by delta_zNear) 
    [↓]     zoom by increasing zNear (by delta_zNear)
    [q/ESC] Quit

"""

# from Users.horiikengo.Documents.Python.dualProjection.realsense_distanse import SCREEN, THRESHOLD
import sys, os
import math
import numpy as np
import cv2
from numpy.core.fromnumeric import nonzero
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from PIL import Image
import pyglet
import pyglet.gl as gl

import ctypes
from plotter import Plotter

#===============================
# 定数
#===============================

TARGET_SCREEN_ID = 1     # プロジェクタのスクリーンID
CAMERA_ID = 1           # カメラID

DATA_DIRNAME = "data"
DATA_DIRPATH = os.path.join(os.path.dirname(__file__), DATA_DIRNAME)
if not os.path.exists(DATA_DIRPATH):
    os.makedirs(DATA_DIRPATH)

CHESS_HNUM = 16       # 水平方向個数
CHESS_VNUM = 9      # 垂直方向個数
CHESS_MARGIN = 0    # [px]h
CHESS_BLOCKSIZE = 80 # [px]

BOARD_WIDTH  = 0.8  # chessboard の横幅[m]
BOARD_HEIGHT = 0.45  # chessboard の縦幅 [m]
BOARD_X = 0.         # chessboard の3次元位置X座標 [m]（右手系）
BOARD_Y = 0.         # chessboard の3次元位置Y座標 [m]（右手系）
BOARD_Z = -1.6       # chessboard の3次元位置Z座標 [m]（右手系）[see]

DEPTH_LIMIT = 2.0    # 点群処理をする最大の距離 [m]
MAX_PLANE_NUM = 3    # シーンから検出する平面の最大数

LARGE_VALUE = 99999      # 最適化の初期値（再投影誤差）
error_min = LARGE_VALUE  # 最適化の最良値（再投影誤差）




# OpenGL の射影のパラメータ
class Params:
    def __init__(self, zNear = 0.0001, zFar = 20.0, fovy = 20.0):
        self.Z_NEAR = zNear     # 最も近い点 [m]
        self.Z_FAR  = zFar      # 最も遠い点 [m]
        self.FOVY   = fovy      # 縦の視野角 [deg]


PARAMS = Params(zNear = 0.0001, # [m]
                zFar = 20.0,    # [m]
                fovy = 20.0     # [deg]
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
board_vertices = ((BOARD_X - BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT / 2, BOARD_Z),
                (BOARD_X - BOARD_WIDTH / 2, BOARD_Y - BOARD_HEIGHT / 2, BOARD_Z),
                (BOARD_X + BOARD_WIDTH / 2, BOARD_Y - BOARD_HEIGHT / 2, BOARD_Z),
                (BOARD_X + BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT / 2, BOARD_Z))

camera = None

TARGET_HNUM = 5       # チェッカーボード上のマーカ個数（水平方向）
TARGET_VNUM = 5       # チェッカーボード上のマーカ個数（垂直方向）
TARGET_DIAMETER = 10  # チェッカーボード上のマーカサイズ（直径）

# [see] 3次元標定点（マーカ）の設定
POINTS_3D_CSV = 'points3d_1.csv'
CORRESPONDENCES_CSV_PATH = os.path.join(DATA_DIRPATH, POINTS_3D_CSV)

# [see] 最適化結果の保存先ファイル
PARAMS_FILE = 'parameters.npz'
PARAMS_FILE_PATH = os.path.join(DATA_DIRPATH, PARAMS_FILE)

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
        self.translation = np.array([0, 0, 0], np.float32)
        self.distance = 0
        self.paused = False

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

        self.half_fov = False                  # プロジェクタの画角の変数
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
# 回転に関する関数
#===============================
# https://stackoverflow.com/a/6802723
# def rotation_matrix(axis, theta):
#     """
#     Return the rotation matrix associated with counterclockwise rotation about
#     the given axis by theta radians.
#     """
#     axis = np.asarray(axis)
#     axis = axis / math.sqrt(np.dot(axis, axis))
#     a = math.cos(theta / 2.0)
#     b, c, d = -axis * math.sin(theta / 2.0)
#     aa, bb, cc, dd = a * a, b * b, c * c, d * d
#     bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
#     return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
#                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
#                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


# [see] 外因性オイラー角でのロール・ピッチ・ヨーから回転行列への変換
def rotation_matrix_rpy_euler(roll, pitch, yaw):
    sr = np.sin(roll)
    sp = np.sin(pitch)
    sy = np.sin(yaw)
    cr = np.cos(roll)
    cp = np.cos(pitch)
    cy = np.cos(yaw)

    rm = (gl.GLfloat * 16)()
    rm[0] = sp*sr*sy + cr*cy
    rm[1] = sr*cp
    rm[2] = sp*sr*cy - sy*cr
    rm[3] = 0
    rm[4] = sp*sy*cr - sr*cy
    rm[5] = cp*cr
    rm[6] = sp*cr*cy + sr*sy
    rm[7] = 0
    rm[8] = sy*cp
    rm[9] = -sp
    rm[10] = cp*cy
    rm[11] = 0
    rm[12] = 0
    rm[13] = 0
    rm[14] = 0
    rm[15] = 1
    return rm


# 回転行列から外因性オイラー角でのロール・ピッチ・ヨーへの変換
def rotation_matrix_2_rpy_euler_angle(R, delta=1e-5):
    roll, pitch, yaw = 0, 0, 0
    if np.abs(R[1, 2] + 1.0) < delta:  # sin(alpha) == -1 の場合 beta と gamma は一意に定まらないので，beta = 0 とする．
        roll = np.arctan2(R[2, 0], R[2, 1])
        pitch = - np.pi / 2.0
        yaw = 0.0
    elif np.abs(R[1, 2] - 1.0) < delta:  # sin(alpha) == 1 の場合 beta と gamma は一意に定まらないので，beta = 0 とする．
        roll = np.arctan2(R[2, 0], R[2, 1])
        pitch = np.pi / 2.0
        yaw = 0.0
    else:
        roll = np.arctan2(R[1, 0], R[1, 1])
        pitch  = -np.arcsin(R[1, 2])
        yaw = np.arctan2(R[0, 2], R[2, 2])

    return roll, pitch, yaw


#===============================
# 関数群
#===============================
# copy our data to pre-allocated buffers, this is faster than assigning...
# pyglet will take care of uploading to GPU
def copy(dst, src):
    """copy numpy array to pyglet array"""
    # timeit was mostly inconclusive, favoring slice assignment for safety
    np.array(dst, copy=False)[:] = src.ravel()


def make_chessboard(num_h, num_v, margin, block_size):
    chessboard = np.ones((block_size * num_v + margin * 2, block_size * num_h + margin * 2, 3), dtype=np.uint8) * [255, 165, 0][::-1]
    
    for y in range(num_v):
        for x in range(num_h):
            if (x + y) % 2 == 0:
                sx = x * block_size + margin
                sy = y * block_size + margin
                chessboard[sy:sy + block_size, sx:sx + block_size, 0] = 0
                chessboard[sy:sy + block_size, sx:sx + block_size, 1] = 0
                chessboard[sy:sy + block_size, sx:sx + block_size, 2] = 0

    return chessboard


# チェスボード上に標定点の追加
def add_control_points(chessboard, num_h, num_v):

    if num_h%2 == 0 or num_v%2 == 0 or num_h < 3 or num_v < 3:
        print("error: num_h and num_v should be odd number >= 3 @ (num_h, num_v) = ({},{})".format(num_h, num_v))

    _chessboard = chessboard.copy()
    h, w, _ = _chessboard.shape
    dh = h / (num_v - 1)
    dw = w / (num_h - 1)
    for y in range(num_v):
        ty = int(dh * y)
        for x in range(num_h):
            if x == 0 or x == num_h-1 or y == 0 or y == num_v-1:
                tx = int(dw * x)
                _chessboard = cv2.circle(_chessboard, (tx, ty), TARGET_DIAMETER, (0, 0, 255), -1)

    return _chessboard


def load_chessboard():
    global chessboard_image, texture_ids

    chessboard = make_chessboard(CHESS_HNUM, CHESS_VNUM, CHESS_MARGIN, CHESS_BLOCKSIZE)
    chessboard = add_control_points(chessboard, TARGET_HNUM, TARGET_VNUM)

    filepath = os.path.join(DATA_DIRPATH, 'chessboard.png')
    cv2.imwrite(filepath, chessboard)
    chessboard_image = Image.open(filepath)

    tw, th = chessboard_image.width, chessboard_image.height
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_ids[0])
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, tw, th, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, chessboard_image.tobytes())


# [see] 3次元特徴点のロード
def load_global_correspondences(csv_path):
    info = ''
    state.cp3d_opengl = []
    with open(csv_path) as f:
        lines = f.read().split('\n')
        for line in lines:
            if line.startswith('#'):
                info = line[1:]
            elif line:
                if info.startswith('planes'):
                    planes_num = int(line)
                    state.cp3d_opengl = [np.empty((0, 3)) for _ in range(planes_num)]
                elif info.startswith('pid'):
                    data = line.split(',')
                    pid = int(data[0])
                    x = float(data[1])
                    y = float(data[2])
                    z = float(data[3])
                    state.cp3d_opengl[pid] = np.vstack((state.cp3d_opengl[pid], np.array((x, y, z))))


def check_nones(params):
    for p in params:
        if p is None:
            return False
    return True


def save_params():

    if check_nones((state.tvec, state.rvec,
                    state.camera_frame,
                    state.cp3d_opengl, state.cp2d_projected, state.cp2d_cpoint,
                    state.H_pj_cm)):
        np.savez(PARAMS_FILE_PATH,
                rpy = np.array([state.roll, state.pitch, state.yaw]),
                trans = state.tvec,
                rvec = state.rvec,
                _camera_frame = state.camera_frame,
                cp3d_opengl = state.cp3d_opengl,
                cp2d_projected = state.cp2d_projected,
                cp2d_cpoint = state.cp2d_cpoint,
                H_pj_cm = state.H_pj_cm)
        print("Parameters are saved successfully.")
        np.savetxt("homography_pjcm.txt", state.H_pj_cm)
    else:
        print("Error: calibration is not finished, any parameters is not saved.")


def load_params():
    try:
        with np.load(PARAMS_FILE_PATH, allow_pickle=True) as X:
            rpy, state.tvec, state.rvec, \
            state._camera_frame, \
            state.cp3d_opengl, state.cp2d_projected, state.cp2d_cpoint, \
            state.H_pj_cm =\
                [X[i] for i in ('rpy', 'trans', 'rvec',
                                '_camera_frame',
                                'cp3d_opengl',
                                'cp2d_projected', 'cp2d_cpoint',
                                'H_pj_cm')]
            state.roll = rpy[0]
            state.pitch = rpy[1]
            state.yaw = rpy[2]
            print("Parameters are loaded successfully.")
    except:
        print("Parameters are not loaded, something has happened while loading parameters.")


#-------------------------------
# 描画関数
#-------------------------------
def axes(size=1, width=1):
    gl.glMatrixMode(gl.GL_MODELVIEW)

    """draw 3d axes"""
    gl.glLineWidth(width)
    pyglet.graphics.draw(6, gl.GL_LINES,
                        ('v3f', (0, 0, 0, size, 0, 0,
                                0, 0, 0, 0, size, 0,
                                0, 0, 0, 0, 0, size)),
                        ('c3f', (1, 0, 0, 1, 0, 0,
                                0, 1, 0, 0, 1, 0,
                                0, 0, 1, 0, 0, 1,
                                ))
                        )


# 地面のグリッドの描画
def grid(size=1, n=10, width=1):
    gl.glMatrixMode(gl.GL_MODELVIEW)

    """draw a grid on xz plane"""
    gl.glLineWidth(width)
    s = size / float(n)
    s2 = 0.5 * size
    batch = pyglet.graphics.Batch()

    for i in range(0, n + 1):
        x = -s2 + i * s
        batch.add(2, gl.GL_LINES, None, ('v3f', (x, 0, -s2, x, 0, s2)))
    for i in range(0, n + 1):
        z = -s2 + i * s
        batch.add(2, gl.GL_LINES, None, ('v3f', (-s2, 0, z, s2, 0, z)))

    batch.draw()


def board():
    global chessboard_image, texture_ids
    
    gl.glMatrixMode(gl.GL_MODELVIEW)

    gl.glEnable(gl.GL_TEXTURE_2D)
    gl.glBindTexture(gl.GL_TEXTURE_2D, texture_ids[0])
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    gl.glTexEnvi(gl.GL_TEXTURE_ENV, gl.GL_TEXTURE_ENV_MODE, gl.GL_REPLACE)
    
    gl.glMatrixMode(gl.GL_TEXTURE)
    gl.glPushMatrix()
    gl.glLoadIdentity()
    gl.glTranslatef(0.5 / chessboard_image.width, 0.5 / chessboard_image.height, 0)

    gl.glBegin(gl.GL_QUADS)
    gl.glTexCoord2i(0, 0)
    gl.glVertex3f(*board_vertices[0])
    gl.glTexCoord2i(0, 1)
    gl.glVertex3f(*board_vertices[1])
    gl.glTexCoord2i(1, 1)
    gl.glVertex3f(*board_vertices[2])
    gl.glTexCoord2i(1, 0)
    gl.glVertex3f(*board_vertices[3])
    gl.glEnd ()
    gl.glPopMatrix()

    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    gl.glDisable(gl.GL_TEXTURE_2D)

#-------------------------------
# ここからイベント関数
#-------------------------------
def on_mouse_drag_impl(x, y, dx, dy, buttons, modifiers):
    cam_w, cam_h = map(float, window.get_size())

    if buttons & pyglet.window.mouse.LEFT:
        state.yaw -= dx * 0.001
        state.pitch += dy * 0.001

    if buttons & pyglet.window.mouse.RIGHT:
        # dp = np.array((dx / cam_w, -dy / cam_h, 0), np.float32)
        # state.translation += np.dot(state.rotation, dp)
        state.tvec += np.array((dx, dy, 0)) * 0.002

    if buttons & pyglet.window.mouse.MIDDLE:
        state.roll -= dx * 0.001


def on_mouse_button_impl(x, y, button, modifiers):
    state.mouse_btns[0] ^= (button & pyglet.window.mouse.LEFT)
    state.mouse_btns[1] ^= (button & pyglet.window.mouse.RIGHT)
    state.mouse_btns[2] ^= (button & pyglet.window.mouse.MIDDLE)


def on_mouse_scroll_impl(x, y, scroll_x, scroll_y):
    dz = scroll_y * 0.1
    state.tvec[2] += dz


def current_opt_func_name():
    print("Current method for optimization is")
    if state.opt_func & AppState.OPT_SIMPLEX:
        print("\t Simplex by scipy.optimize.fmin()")
    if state.opt_func & AppState.OPT_LM:
        print("\t Levenberg-Marquardt by scipy.optimize.leastsq()")
    if state.opt_func & AppState.OPT_BASINHOPPING:
        print("\t Basinhopping by scipy.optimize.basinhopping()")
    if state.opt_func & AppState.OPT_BRUTE:
        print("\t Brute-Force by scipy.optimize.brute()")


# [key]
def on_key_press_impl(symbol, modifiers):

    if symbol == pyglet.window.key._1:
        state.opt_func = AppState.OPT_SIMPLEX
        current_opt_func_name()

    if symbol == pyglet.window.key._2:
        state.opt_func = AppState.OPT_LM
        current_opt_func_name()

    if symbol == pyglet.window.key._3:
        state.opt_func = AppState.OPT_BASINHOPPING
        current_opt_func_name()

    if symbol == pyglet.window.key._4:
        state.opt_func = AppState.OPT_BRUTE
        current_opt_func_name()

    if symbol == pyglet.window.key.F1:
        state.draw_axes = False
        state.draw_grid = False

    # フルスクリーン/ウインドウ表示切り替え
    if symbol == pyglet.window.key.F2:
        if not window.fullscreen:
            # フルスクリーン表示に設定
            window.set_fullscreen(fullscreen=True)
        else:
            # ウインドウ表示に設定
            window.set_fullscreen(fullscreen=False)

    if symbol == pyglet.window.key.A:
        state.draw_axes ^= True
        state.draw_grid ^= True

    if symbol == pyglet.window.key.B:
        state.draw_board ^= True

    if symbol == pyglet.window.key.C:
        on_draw_impl()
        state.set_control_points = True
   
    if symbol == pyglet.window.key.F:
        state.half_fov ^=True

    if symbol == pyglet.window.key.G:
        state.draw_grid ^= True

    if symbol == pyglet.window.key.H:
        on_draw_impl()
        state.obtain_homography_matrix = True

    if symbol == pyglet.window.key.I:
        show_Rt()

    # [debug]
    if symbol == pyglet.window.key.J:
        on_draw_impl()
        if state.cp3d_opengl is not None and state.H_pj_cm is not None:
            get_camera_frame()
            pid = 0
            check_point3d_on_frame(state.cp3d_opengl[pid], state.camera_frame, state.H_pj_cm, window.get_size())

    # [debug]
    if symbol == pyglet.window.key.K:
        on_draw_impl()
        if state.cp3d_opengl is not None and state.cp2d_cpoint is not None and state.H_pj_cm is not None:
            get_camera_frame()
            pid = 0
            check_H_pj_cm(state.cp3d_opengl[pid], state.H_pj_cm, points2d_overlay=state.cp2d_cpoint[pid])

    if symbol == pyglet.window.key.O:
        optimize()

    # [debug]
    if symbol == pyglet.window.key.P:
        on_draw_impl()
        check_control_points()

    if symbol == pyglet.window.key.Q:
        window.close()

    if symbol == pyglet.window.key.R:
        state.reset()

    if symbol == pyglet.window.key.S:
        save_screen()

    if symbol == pyglet.window.key.T:
        pyglet.clock.schedule(run_realsense)

    # if symbol == pyglet.window.key.X:
    #     window.set_fullscreen(fullscreen=False)
    #     plotter = Plotter()
    #     plotter.set_callback(myCallback())
    #     plotter.show(adjust=False)
    #     move_plotter()

    if symbol == pyglet.window.key.X:
        load_params()

    if symbol == pyglet.window.key.Z:
        save_params()
        np.savetxt("modelview_matrix.txt", modelview_matrix)
        np.savetxt("projection_matrix.txt", projection_matrix)

    if symbol == pyglet.window.key.UP:
        state.zNear += state.delta_zNear
        print("current zNear = ", state.zNear)

    if symbol == pyglet.window.key.DOWN:
        state.zNear -= state.delta_zNear
        while state.zNear < 0:
            state.zNear += state.delta_zNear
        print("current zNear = ", state.zNear)

    if symbol == pyglet.window.key.RIGHT:
        state.delta_zNear *= 2
        print("current delta_zNear = ", state.delta_zNear)

    if symbol == pyglet.window.key.LEFT:
        state.delta_zNear /= 2
        print("current delta_zNear = ", state.delta_zNear)


#-------------------------------
# ここから座標変換用の関数
#-------------------------------
def projection():
    global projection_matrix

    width, height = window.get_size()
    gl.glViewport(0, 0, width, height)

    # 射影行列の設定
    gl.glMatrixMode(gl.GL_PROJECTION)
    gl.glLoadIdentity()

    #-----------------
    # 従来法
    #-----------------
    # aspect = width / float(height * 2)
    # bottom = 0
    # top = state.zNear * np.tan(np.radians(PARAMS.FOVY))
    # left = - top * aspect
    # right = top * aspect
    # gl.glFrustum(left, right, bottom, top, state.zNear, PARAMS.Z_FAR)
    #
    # pm = (gl.GLfloat * 16)()
    # gl.glGetFloatv(gl.GL_PROJECTION_MATRIX, pm)
    # print(list(pm))
    #
    # pm[0] = 2 * state.zNear / (right - left)
    # pm[5] = 2 * state.zNear / (top - bottom)
    # pm[8] = (right + left) / (right - left)
    # pm[9] = (top + bottom) / (top - bottom)
    # pm[10] = - (PARAMS.Z_FAR + state.zNear) / (PARAMS.Z_FAR - state.zNear)
    # pm[11] = - 1
    # pm[14] = - 2 * PARAMS.Z_FAR * state.zNear / (PARAMS.Z_FAR - state.zNear)
    # print(list(pm))

    #-----------------
    # [see] 新手法
    #-----------------
    fov = PARAMS.FOVY*0.5
    aspect = width / float(height)
    top = state.zNear * np.tan(np.radians(fov))
    bottom = -state.zNear * np.tan(np.radians(fov))
    left = - top * aspect
    right = top * aspect

    pm = (gl.GLfloat * 16)()
    if state.half_fov: #
        pm[0] = 4 * state.zNear / (right - left)
        pm[5] = 4 * state.zNear / (top - bottom)
        pm[8] = (right + left) / (right - left)
        pm[9] = 1 + 2 * (top + bottom) / (top - bottom)
        pm[10] = - (PARAMS.Z_FAR + state.zNear) / (PARAMS.Z_FAR - state.zNear)
        pm[11] = - 1
        pm[14] = - 2 * PARAMS.Z_FAR * state.zNear / (PARAMS.Z_FAR - state.zNear)
    else:
        pm[0] = 2 * state.zNear / (right - left)
        pm[5] = 2 * state.zNear / (top - bottom)
        pm[8] = (right + left) / (right - left)
        pm[9] = (top + bottom) / (top - bottom)
        pm[10] = - (PARAMS.Z_FAR + state.zNear) / (PARAMS.Z_FAR - state.zNear)
        pm[11] = - 1
        pm[14] = - 2 * PARAMS.Z_FAR * state.zNear / (PARAMS.Z_FAR - state.zNear)
    gl.glLoadMatrixf((ctypes.c_float * 16)(*pm))

    projection_matrix = np.array(pm).reshape(4,4).transpose()
    # CG内の行列
    # print("projection:{}".format(projection_matrix))
    # # f = open("projection_matrix.txt","w")
    # # f.write("{}".format(projection_matrix))
    # np.savetxt("projection_matrix.txt", projection_matrix)
    # print("CG内の内部パラメータOK")

    # f.close()


def modelview():
    global modelview_matrix

    gl.glMatrixMode(gl.GL_MODELVIEW)

    gl.glLoadIdentity()

    _matrix = rotation_matrix_rpy_euler(state.roll, state.pitch, state.yaw)
    _matrix[12] = state.tvec[0]
    _matrix[13] = state.tvec[1]
    _matrix[14] = state.tvec[2]
    _matrix[15] = 1
    gl.glLoadMatrixf((ctypes.c_float * 16)(*_matrix))

    # 視点の設定（デフォルトの座標系なのでなくても同じ）
    # gluLookAt( float eyeX, float eyeY, float eyeZ,
    #            float centerX, float centerY, float centerZ,
    #            float upX, float upY, float upX
    # )
    gl.gluLookAt(0.0, 0.0, 0.0,
                0.0, 0.0, -1.0,
                0.0, 1.0, 0.0)

    #-------------------------------------
    # numpy 配列への変換
    mm = (gl.GLfloat * 16)()
    gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX, mm)
    modelview_matrix = np.array(mm).reshape(4, 4).transpose()
    # print("modelview:{}".format(modelview_matrix))
    # f = open("modelview_matrix.txt","w")
    # f.write("{}".format(modelview_matrix))
    # f.close()
    # np.savetxt("modelview_matrix.txt", modelview_matrix)
    #-------------------------------------
    # 回転ベクトルへの変換
    R = modelview_matrix[0:3, 0:3]
    rvec, _ = cv2.Rodrigues(R)
    state.rvec = rvec.reshape(3)
    # print(state.rvec)

    # #-------------------------------------
    # # roll (z), pitch (x), yaw (y) 角への変換
    # state.roll, state.pitch, state.yaw = rotation_matrix_2_rpy_euler_angle(state.R)


# [see] CG 内の3次元座標を画像へ投影
def world2cam(point3d, mv_matrix, pj_matrix, img_size):
    """
    :param point3d: 世界座標系での三次元座標
    :param mv_matrix:
    :param pj_matrix:
    :param img_size:
    :return:
    """
    _point3d = np.hstack((point3d, 1))
    _point2d = np.dot(mv_matrix, _point3d)
    _point2d = np.dot(pj_matrix, _point2d)

    # 同次座標の不定性の解消（s_1 = _point2d[2]）
    (w, h) = img_size
    # print("wh",img_size)
    u = (int)(w / 2 * (_point2d[0] / _point2d[2] + 1))
    v = (int)(h - h / 2 * (_point2d[1] / _point2d[2] + 1))

    return (u, v)


# ある点についてのホモグラフィによる変換先の座標を取得
def homography_by_point(point, H, src_size):
    (w, h) = src_size
    uv0 = world2cam(point, modelview_matrix, projection_matrix, (w, h))
    uv = np.dot(H, np.array([uv0[0], uv0[1], 1]))
    uv /= uv[2]
    return uv, uv0


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

    # hidariue = (BOARD_X - BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT, BOARD_Z)
    # w, h = window.get_size()
    # # print("w, h", w, h)
    # (u, v) = world2cam(hidariue, modelview_matrix, projection_matrix, (w, h))
    # # (u, v) = world2cam(p, modelview_matrix, projection_matrix, (w, h))
    # # print("uvuv", u,v)
    # img = cv2.imread("q.png")
    # img = cv2.circle(img, (u, v), 15, (0, 255, 0), thickness=-1)
    # cv2.imwrite("p.png", img)

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


def overlay_points2d_on_frame(points2d, frame):
    img_pj = frame.copy()
    for p in points2d:
        u = int(p[0])
        v = int(p[1])
        img_pj = cv2.circle(img_pj, (u, v), 5, (0, 0, 255), 2)
    return img_pj


#-------------------------------
# ここからコマンド関数
#-------------------------------
def obtain_control_point():
    check_control_points()

    if get_camera_frame():
        # print("test", state.camera_frame)
        # cv2.imshow("title",state.camera_frame)
        # print("obtain")
        state.cp2d_cpoint = [np.empty((0, 2)) for _ in range(len(state.cp3d_opengl))]

        pid = 0  # [see] まずは1つの平面のみを実装
        window.set_fullscreen(fullscreen=False)
        # plotter = Plotter(imgarray=state.camera_frame)
        camera_frame = cv2.imread("./align/align_img_c4.png")
        camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)
        plotter = Plotter(imgarray=camera_frame)
        # plotter = np.array(Image.open('./align/align_img_c.png'))
        move_plotter()

        class _callback(Plotter.Callback):
            def on_quit(self):
                # print("on_quit")
                plt.close('all')
                window.set_fullscreen(fullscreen=True)
                state.cp2d_cpoint[pid] = plotter.GetImagePointsArray()
                print("------------------")
                print("get {} points".format(len(state.cp2d_cpoint[pid])))
                # print(state.cp2d_cpoint[pid])
                # print("------------------")

        plotter.SetCallback(_callback())
        plotter.show(adjust=False)

        p3ds_cg = state.cp3d_opengl[pid]
        p2ds_sr = state.cp2d_cpoint[pid]

        if len(p3ds_cg) == len(p2ds_sr):
            img_pj = overlay_points2d_on_frame(state.cp2d_cpoint[pid], state.camera_frame)
            preview(img_pj)
   

def obtain_homography_matrix():
    check_control_points()

    if get_camera_frame():
        state.cp2d_projected = [np.empty((0, 2)) for _ in range(len(state.cp3d_opengl))]

        pid = 0  # [see] まずは1つの平面のみを実装

        # plotter = Plotter(imgarray=state.camera_frame)
        # plotter = np.array(Image.open('./align/align_img_h.png'))
        camera_frame = cv2.imread("./align/align_img_h4.png")
        camera_frame = cv2.cvtColor(camera_frame, cv2.COLOR_RGB2BGR)
        plotter = Plotter(imgarray=camera_frame)
        move_plotter()
        class _callback(Plotter.Callback):
            def on_quit(self):
                plt.close('all')
                window.set_fullscreen(fullscreen=True)
                state.cp2d_projected[pid] = plotter.GetImagePointsArray()
                print("------------------")
                print("get {} points".format(len(state.cp2d_projected[pid])))
                # print(state.cp2d_projected[pid])
                # print("------------------")
        plotter.SetCallback(_callback())
        plotter.show(adjust=False)

        # window.set_fullscreen(fullscreen=False)

        # プロジェクション ＋ カメラ撮影のホモグラフィ行列の推定
        H = estimate_homography(pid)
        # print("homography_pjcm:{}".format(H))
        # inv_H = np.linalg.inv(H)
        # f = open("homography_pjcm.txt","w")
        # f.write("{}".format(inv_H))
        # f.close()
        # np.savetxt("homography_pjcm.txt", H)

        if H is not None:
            state.H_pj_cm = H
            check_H_pj_cm(state.cp3d_opengl[pid], state.H_pj_cm, points2d_overlay=state.cp2d_projected[pid])


# [see] CG画面からスクリーンへのホモグラフィ行列の推定
def estimate_homography(plane_id):
    w, h = window.get_size()

    p3ds_cg = state.cp3d_opengl[plane_id]
    p2ds_pj = state.cp2d_projected[plane_id]

    if len(p3ds_cg) != len(p2ds_pj):
        print("Error: number of correspondence points does not match.")
        return None

    src = np.empty((0, 2))  # CG の3次元モデルを2次元に投影した投影対象画像上の対応点
    dst = p2ds_pj           # カメラで撮影した投影された画像上の対応点
    for p in p3ds_cg:
        (u, v) = world2cam(p, modelview_matrix, projection_matrix, (w, h))
        src = np.vstack([src, np.array([u,v])])

    # H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    # H, mask = cv2.findHomography(src, dst, cv2.LMEDS)
    H, mask = cv2.findHomography(src, dst, 0)

    return H


# [see] 最適化
def optimize():

    plane_id = 0
    if state.cp3d_opengl is None or state.cp2d_cpoint is None:
        print("Error: correspondence is not set.")
        print("Error: correspondence is not set.")
        return None

    _cp3d = state.cp3d_opengl[plane_id]
    _cp2d = state.cp2d_cpoint[plane_id]
    if len(_cp3d) != len(_cp2d):
        print("Error: number of correspondence points does not match between 3d and 2d.")
        return None

    global error_min
    error_min = LARGE_VALUE

    def convert_params(params):
        state.tvec = np.array(params[0:3])
        state.rvec = np.array(params[3:6]).reshape(3, 1)
        R, _ = cv2.Rodrigues(state.rvec[0:3])

        # roll (z), pitch (x), yaw (y) 角への変換
        state.roll, state.pitch, state.yaw = rotation_matrix_2_rpy_euler_angle(R)


    def error_func(params, cp3d, cp2d):
        trans = np.array(params[0:3])
        rvec = np.array(params[3:6]).reshape(3, 1)

        R, _ = cv2.Rodrigues(rvec)
        _modelview_matrix = np.zeros((4,4))
        _modelview_matrix[0:3, 0:3] = R
        _modelview_matrix[0,3] = trans[0]
        _modelview_matrix[1,3] = trans[1]
        _modelview_matrix[2,3] = trans[2]
        _modelview_matrix[3,3] = 1

        error = np.zeros(len(cp3d))
        w, h = window.get_size()
        for i, p3 in enumerate(cp3d):
            c2uv = cp2d[i]
            (c3u, c3v) = world2cam(p3, _modelview_matrix, projection_matrix, (w, h))
            c3uv = np.dot(state.H_pj_cm, np.array([c3u, c3v, 1]))
            c3uv /= c3uv[2]
            error[i] = np.linalg.norm(c3uv[0:2] - c2uv)
            # print(error[i])

        global error_min
        current_error = np.linalg.norm(error) / len(error)
        if (current_error - error_min).any():
            error_min = current_error
            if not state.opt_func & AppState.OPT_BRUTE:
                print("updated: reprojection error = {} [px/point]".format(current_error, " @ ", params))

        if state.opt_func & AppState.OPT_LM:
            return error
        else:
            return current_error

    def callback(params):
        convert_params(params)
        on_draw_impl()

    initial_params = [*state.tvec, *state.rvec]

    if state.opt_func & AppState.OPT_SIMPLEX:
        optimized_params = opt.fmin(error_func, initial_params, args=(_cp3d, _cp2d), callback=callback, xtol=0.000001, ftol=0.000001)
    if state.opt_func & AppState.OPT_LM:
        optimized_params, _ = opt.leastsq(error_func, x0=initial_params, args=(_cp3d, _cp2d), epsfcn=10)
    if state.opt_func & AppState.OPT_BASINHOPPING:
        minimizer_kwargs = {"args": (_cp3d, _cp2d)}
        res = opt.basinhopping(error_func, initial_params, minimizer_kwargs=minimizer_kwargs)
        optimized_params = res.x
        print(res)
        print("optimized error = ", res.fun)
    if state.opt_func & AppState.OPT_BRUTE:
        def pslice(p, rp):
            return slice(p-rp, p+rp)
        ps = initial_params
        range_t = 0.1   # [m]
        range_r = 1 * np.pi / 180  # [rad]
        ranges = (pslice(ps[0], range_t),
                pslice(ps[1], range_t),
                pslice(ps[2], range_t),
                pslice(ps[3], range_r),
                pslice(ps[4], range_r),
                pslice(ps[5], range_r))
        optimized_params = opt.brute(error_func, ranges=ranges, Ns=50, args=(_cp3d, _cp2d), disp=True, finish=opt.fmin)

    convert_params(optimized_params)
    on_draw_impl()

    print("-------------------")
    print(initial_params)
    print(optimized_params)


#-------------------------------
# ここから機能関数
#-------------------------------
def get_camera_frame():
    global camera

    ret = False
    if camera is None:
        camera = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)

    if camera:
        _, state.camera_frame = camera.read()
        print(state.camera_frame.shape)
        print("[real camera] Captured")
        if state.camera_frame is not None:
            state.camera_frame = cv2.cvtColor(state.camera_frame, cv2.COLOR_RGB2BGR)
        ret = True

    if camera:
        camera.release()
        camera = None

    return ret


def get_screen_capture():
    # pyglet.image.get_buffer_manager().get_color_buffer().save('out.png')
    scdata = pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
    format = "RGBA"
    pitch = scdata.width * len(format)
    img = np.asanyarray(scdata.get_data(format, pitch))

    width, height = window.get_size()
    img = img.reshape(height, width, -1)
    img = cv2.flip(img, 0)
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    return img


def move_plotter(offset=(25, 50)):
    ww, wh = window.get_size()
    wx, wy = window.get_location()

    fm = plt.get_current_fig_manager()
    geom = fm.window.geometry()
    x, y, dx, dy = geom.getRect()

    if wx>0:
        fm.window.setGeometry(offset[0], offset[1], dx, dy)
    else:
        fm.window.setGeometry(ww+offset[0], offset[1], dx, dy)
    fm.window.setFocus()


def preview(img, offset=(25,50), title=None):
    plt.figure()
    move_plotter(offset)
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.1)


def save_screen():
    img = get_screen_capture()
    preview(img)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("out.png", img)


def show_Rt():
    print("=================================================")
    print("Camera info.")
    print("=================================================")
    print("Position")
    print("\t(x, y, z) = ({:.3f}, {:.3f}, {:.3f}) [m]".format(*state.tvec))
    print("extrinsic zxy-euler angle (roll, pitch, yaw) ")
    rpy = np.array((state.roll, state.pitch, state.yaw))
    print("\t(roll, pitch, yaw) = ({:.3f}, {:.3f}, {:.3f}) [rad]".format(*rpy))
    print("\t(roll, pitch, yaw) = ({:.3f}, {:.3f}, {:.3f}) [deg]".format(*(rpy*180/np.pi)))
    print("=================================================")

#-------------------------------
# 目視デバッグ用関数
#-------------------------------
# [debug] 推定したホモグラフィ行列の確認
def check_H_pj_cm(points3d, H, points2d_overlay=None):
    img_cg = get_screen_capture()
    img_pj = state.camera_frame

    if points2d_overlay is not None:
        img_pj = overlay_points2d_on_frame(points2d_overlay, state.camera_frame)

    img_warped = img_cg
    img_blended = None
    if H is not None:
        h, w, _ = state.camera_frame.shape
        img_warped = cv2.warpPerspective(img_cg, H, (w, h))
        img_blended = cv2.addWeighted(src1=img_pj, alpha=0.6, src2=img_warped, beta=0.4, gamma=0)

        # [蛇足] 標定点が移っているかの確認（H行列の使い方の確認）
        if points3d is not None:
            for p in points3d:
                uv, uv0 = homography_by_point(p, H, window.get_size())
                img_cg = cv2.circle(img_cg, (int(uv0[0]), int(uv0[1])), 10, [255, 255, 0], -1)
                img_warped = cv2.circle(img_warped, (int(uv[0]), int(uv[1])), 3, [255, 255, 0], -1)
                img_blended = cv2.circle(img_blended, (int(uv[0]), int(uv[1])), 3, [200, 200, 0], -1)


    fig = plt.figure(figsize=(16, 16))
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(img_cg)
    ax1.set_title("[1] from virtual cam.\n (yellow points: control points.)")
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(img_pj)
    ax2.set_title("[2] image from real cam.\n (blue circles: clicked pos.)")
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(img_warped)
    ax3.set_title("[3] warped virtual-cam image [0] by homography.\n (yellow points: control points projected by homography)")
    if img_blended is not None:
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.imshow(img_blended)
        ax4.set_title("[4] composed image of [2] and [3]. ")
    plt.show()


# [see] 標定点の確認
def check_control_points():
    img = get_screen_capture()
    h, w, _ = img.shape

    v = 220
    colors = np.array([(v, 0, 0),
                    (0, v, 0),
                    (0, 0, v),
                    (v, v, 0),
                    (v, 0, v),
                    (0, v, v),
                    (v, v, v)])
    colors = np.vstack((colors, colors/2))
    du, dv = 15, -15
    for ps in state.cp3d_opengl:
        pid = 0
        for p in ps:
            (u, v) = world2cam(p, modelview_matrix, projection_matrix, (w, h))
            cid = pid % len(colors)
            img = cv2.circle(img, (u, v), 15, colors[cid], -1)
            cv2.putText(img, '{}'.format(pid), (u+du, v+dv), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[cid], 3, cv2.LINE_AA)
            pid += 1

    preview(img, title="Control points")


# [debug]
def check_point3d_on_frame(points3d, frame, H, src_size):
    img_pj = frame.copy()
    for p in points3d:
        uv, _ = homography_by_point(*p.reshape(1, 3), H, src_size)
        img_pj = cv2.circle(img_pj, (int(uv[0]), int(uv[1])), 3, [255, 0, 0], -1)
    preview(img_pj)






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

    @window.event
    def on_key_press(symbol, modifiers):
        on_key_press_impl(symbol, modifiers)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        on_mouse_drag_impl(x, y, dx, dy, buttons, modifiers)

    @window.event
    def on_mouse_scroll(x, y, scroll_x, scroll_y):
        on_mouse_scroll_impl(x, y, scroll_x, scroll_y)

    @window.event
    def on_mouse_press(x, y, button, modifiers):
        on_mouse_button_impl(x, y, button, modifiers)

    @window.event
    def on_mouse_release(x, y, button, modifiers):
        on_mouse_button_impl(x, y, button, modifiers)




    #------------------------------
    # OpenGL 用の変数の準備
    #------------------------------
    # チェスボードの作成
    texture_ids = (pyglet.gl.GLuint * 1) ()
    gl.glGenTextures(1, texture_ids)
    load_chessboard()
    load_global_correspondences(CORRESPONDENCES_CSV_PATH)

    # Start
    pyglet.app.run()

    if camera is not None:
        camera.release()
        cv2.destroyAllWindows()




    # try:
    #     # pygletのなにか？
    #     pyglet.app.run()
    # finally:
    #     pipeline.stop()
