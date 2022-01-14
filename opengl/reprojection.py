# https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl
# pip install PyOpenGL
# pip install glfw
# pip install scipy

import math
import numpy as np
from scipy.spatial.transform import Rotation
import cv2
import matplotlib.pyplot as plt

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import glfw
import ctypes

from scipy.spatial import ConvexHull

from camera import GLCamera
from primitive import Cube, Sphere, Polygon

#---------------------------
# Global 変数
DISPLAY_TITLE  = "Reprojection Sample"
DEFAULT_DISPLAY_WIDTH  = 1920
DEFAULT_DISPLAY_HEIGHT = 1080

display_width  = DEFAULT_DISPLAY_WIDTH
display_height = DEFAULT_DISPLAY_HEIGHT

stop_flag = False

objects = []

cameras = {}
cv_cam = None

# objects
wall = None
floor = None
cube = None

def main():
    # Initialize glfw
    if not glfw.init():
        raise RuntimeError('Could not initialize GLFW3')

    cameras['main_view'] = GLCamera(display_size=[display_width, display_height], display_title="main camera view")
    cameras['main_view'].on_keyboard_event.append(on_keyboard_event)
    cameras['main_view'].on_draw.append(windows_refresh)

    cameras['sub_view'] = GLCamera(display_size=[display_width, display_height], display_title="sub camera view")
    cameras['sub_view'].on_draw.append(windows_refresh)

    # OpenGL version
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 0)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.swap_interval(1)

    # initialization
    initialize()

    def can_draw():
        ret = True
        for c in cameras.values():
            ret &= c.is_available()
            if not ret:
                break
        return ret

    while can_draw():
        plot_points_on_wall(cameras['main_view'], cube.vertices)
        glfw.wait_events()
        if stop_flag:
            break

    # Proper shutdown
    glfw.terminate()


def initialize():
    global wall, floor, cube
    show_graphics_info()

    wall  = Cube(scale=(5.0, 2.0, 1.0), position=(0.0, 1.0, -5.5), diffuse=(0.9, 0.5, 0.9))
    floor = Cube(scale=(5.0, 0.2, 2.0), position=(0.0, -0.1, -4.0), diffuse=(0.9, 0.5, 0.9))
    cube  = Cube(scale=(0.5, 0.5, 0.5), position=(0.0, 0.25, -4.0), diffuse=(1.0, 0.0, 0.0))

    objects.append(wall)
    objects.append(floor)
    objects.append(cube)

    for v in cube.vertices:
        objects.append(Sphere(radius=0.01, position=v, diffuse=(1.0, 1.0, 0.0)))

    for c in cameras.values():
        c.objects = {}
        for i, o in enumerate(objects):
            c.objects[i] = o

    windows_refresh()


def windows_refresh():
    for c in cameras.values():
        c.refresh()

def show_graphics_info():
    print('Vendor :', gl.glGetString(gl.GL_VENDOR))
    print('GPU :', gl.glGetString(gl.GL_RENDERER))
    print('OpenGL version :', gl.glGetString(gl.GL_VERSION))


def on_keyboard_event(window, key, scancode, action, mods):
    # print('KB:', key, chr(key), end=' ')
    # if action == glfw.PRESS:
    #     print('press')
    # elif action == glfw.REPEAT:
    #     print('repeat')
    # elif action == glfw.RELEASE:
    #     print('release')
    if action == glfw.RELEASE:
        if key == glfw.KEY_A:
            plot_points_on_wall(cameras['main_view'], cube.vertices)
            windows_refresh()
        if key == glfw.KEY_P:
            img = get_screen_capture(window)
            plot_points_on_img(cameras['main_view'], img, cube.front_vertices)

            plt.figure()
            plt.imshow(img)
            plt.show()
        if key == glfw.KEY_Q:
            global stop_flag
            stop_flag = True


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
    _point3d = np.dot(mv_matrix, _point3d) # カメラ座標系の3次元点
    _point2d = np.dot(pj_matrix, _point3d) # 正規化デバイス座標系での標準視体積内の表現

    # 同次座標の不定性の解消（s_1 = _point2d[2]）
    (w, h) = img_size
    u = (int)(w / 2 * (_point2d[0] / _point2d[2] + 1))
    v = (int)(h - h / 2 * (_point2d[1] / _point2d[2] + 1))

    return (u, v), _point2d


# [see] 画像座標からその3次元対応点への光線ベクトルを計算
def cam2world(uv, mv_matrix, pj_matrix, img_size):
    """
    :param uv: 画像座標（左上が原点）
    :param mv_matrix:
    :param pj_matrix:
    :param img_size:
    :return:
    """

    (w, h) = img_size

    # 正規化デバイス座標系（-1<x<1, -1<y<1, -1<z<1）に変換
    # z は 1 (=far) とする
    _x = 2 * float(uv[0]) / w - 1 # 原点位置の違いの考慮
    _y = 1 - 2*float(uv[1]) / h # 原点位置の違いの考慮
    _z = 1
    _w = 1
    _point2d = np.array([_x, _y, _z, _w]) # 正規化デバイス座標系での標準視体積内の表現

    _point3d = np.dot(np.linalg.inv(pj_matrix), _point2d)
    _cDc2p = _point3d / _point3d[2] # ここはなんだ？？ ビュー座標系のzで割る？
    _cRw = mv_matrix[0:3, 0:3] # 回転行列だけ取り出している
    _cTc2w = mv_matrix[0:3, 3] # 並進だけ取り出している

    return _cDc2p, _cRw, _cTc2w, _point3d

def get_screen_capture(window):
    # フロントバッファを読み込む（デフォルトはバックバッファ）
    gl.glReadBuffer(gl.GL_FRONT)
    image_buffer = gl.glReadPixels(0, 0, display_width, display_height,
                                   gl.GL_RGB,
                                   gl.GL_UNSIGNED_BYTE)
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(display_height, display_width, 3)
    image = cv2.flip(image, 0)
    return image


def plot_points_on_img(cam, img, points):
    ret = None
    for p in points:
        (u, v), _ = world2cam(p, cam.modelview_matrix, cam.projection_matrix, (display_width, display_height))
        ret = cv2.circle(img,
                         center=(u, v),
                         radius=5,
                         color=(255, 255, 0),
                         thickness=3)
    return ret


def plot_points_on_wall(cam, points):

    wTw2p = wall.front_vertices[0] # 背景の壁の前面上の1点について，世界座標系での座標を取得
    test = wall.front_vertices[1]
    wPs = []
    for p in points:

        uv, _ = world2cam(p, cam.modelview_matrix, cam.projection_matrix, (display_width, display_height))
        cDc2p, cRw, cTc2w, _ = cam2world(uv, cam.modelview_matrix, cam.projection_matrix, (display_width, display_height))
        print("cDc2p",cDc2p)
        # projectionかけた後にzで割ったもの, 回転行列, 並進ベクトル

        wRc = cRw.T
        wDc2p = np.dot(wRc, cDc2p[0:3]) # 世界座標系での光線ベクトル （回転行列 × prjection後のuv座標）
        wTw2c = np.dot(wRc, -cTc2w)     # 世界座標系でのカメラの位置ベクトル （回転行列 × 負の並進ベクトル）
        wTc2p = wTw2p - wTw2c           # 世界座標系でのカメラから対象物までの光線ベクトル （世界座標系における壁の1点 - 世界座標系でのカメラの位置ベクトル）

        k = wTc2p[2] / wDc2p[2]         # 壁面までの光線長さ (世界座標系でのカメラから対象物までの光線ベクトル/世界座標系での光線ベクトルz方向)

        wP = wTw2c + k * wDc2p # 光線ベクトルと壁面との交点 (カメラの位置ベクトルに光線の長さをかけた世界座標系での位置ベクトルを足したもの)
        wPs.append(wP)
        # print("aaa",wPs)
        #print(w_wallpoint)
        #objects.append(Sphere(radius=0.01, position=wP, diffuse=(0.2, 0.2, 0.2)))

    points = np.array(wPs)[:, 0:2]
    for_other_prog = points

    hull = ConvexHull(points)
    hull_points = hull.points[hull.vertices]
    hull_points = np.insert(hull_points, 2, wTw2p[2] + 0.01 , axis=1)
    print("point", hull_points)

    # 影をオブジェクトとしてすべてのカメラに追加
    for c in cameras.values():
        c.textures['shadow'] = Polygon(vertices=hull_points, diffuse=(0.2, 0.2, 0.2))
    import time
    #print(points)
    time.sleep(1)

if __name__ == "__main__":
    pass
main()

