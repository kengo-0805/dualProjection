# from cord_calc import BOARD_HEIGHT, BOARD_WIDTH, BOARD_X, BOARD_Y, BOARD_Z
import cv2
import numpy as np
np.set_printoptions(suppress=True)
# from primitive import Cube, Sphere, Polygon
#
# cameras = {}
#
# cameras['main_view'] = GLCamera(display_size=[display_width, display_height], display_title="main camera view")
# cameras['main_view'].on_keyboard_event.append(on_keyboard_event)
# cameras['main_view'].on_draw.append(windows_refresh)
#
# cameras['sub_view'] = GLCamera(display_size=[display_width, display_height], display_title="sub camera view")
# cameras['sub_view'].on_draw.append(windows_refresh)

def homography_cam2pj():
    pj_cord = []
    # カメラ視点でのバウンディングボックスの座標の読み込み
    bb_cord = np.loadtxt("bb_cord.txt")
    if bb_cord is not None:
        print("bb_cord読み込みOK")
        print(bb_cord.reshape([4, 2]))

    # カメラからプロジェクタへの射影変換行列
    homography_pjcm = np.loadtxt("homography_pjcm.txt")
    homography_cmpj = np.linalg.inv(homography_pjcm)
    if homography_cmpj is not None:
        print("cm→pjのホモグラフィOK")
        print(homography_cmpj)


    # BBからPJ視点への変換
    for i in range(4):
        bb_cord_array = np.array([[bb_cord[0][i]], [bb_cord[1][i]], [1.]])
        # print("bb_cord_array", bb_cord_array)
        # print("homography_cmpj", homography_cmpj)
        additional = np.dot(homography_cmpj, bb_cord_array)
        # print("additional", additional)
        pj_cord = np.append(pj_cord, additional)
        # print("座標", pj_cord[i])
        # np.savetxt("pj_cord.txt", pj_cord)
    pj_cord = pj_cord.reshape([4, 3])




    # ホモグラフィから回転並進への分解
    color_intrinsics = np.loadtxt("color_intrinsics.txt")
    _, R, t, _ = cv2.decomposeHomographyMat(homography_cmpj, color_intrinsics)
    print(R,t)

    # cam→pjに座標変換
        # uv→xyzに変換
    for i in range(4):
        bb_cord_array = np.array([[bb_cord[0][i]], [bb_cord[1][i]], [1.]])
        # print("bb_cord_array", bb_cord_array)
        # print("homography_cmpj", homography_cmpj)
        additional = np.dot(homography_cmpj, bb_cord_array)
        # print("additional", additional)
        pj_cord = np.append(pj_cord, additional)
        # print("座標", pj_cord[i])
        # np.savetxt("pj_cord.txt", pj_cord)
    pj_cord = pj_cord.reshape([4, 3])
    return pj_cord

# # objects
# wall = None
# floor = None
# # ここ
# cube = [[-0.25, 0.00, -3.75], [0.25, 0.00, -3.75], [0.25, 0.50, -3.75], [-0.25, 0.50, -3.75]]
# 	# , [0.25, 0.00, -4.25], [-0.25, 0.00, -4.25], [-0.25, 0.50, -4.25], [0.25, 0.50, -4.25]]

DEFAULT_DISPLAY_WIDTH  = 1920
DEFAULT_DISPLAY_HEIGHT = 1080

display_width  = DEFAULT_DISPLAY_WIDTH
display_height = DEFAULT_DISPLAY_HEIGHT


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



def plot_points_on_wall():

    # wTw2p = wall.front_vertices[0] # 背景の壁の前面上の1点について，世界座標系での座標を取得 [-2.5, 0, -5]
    # チェスボードの左上の座標を入れる
    wTw2p = [-2.5, 0, -5]
    BOARD_X = 0.
    BOARD_Y = 0.
    BOARD_Z = -3
    BOARD_WIDTH = 0.8
    BOARD_HEIGHT = 0.45
    board_vertices = ((BOARD_X - BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT, BOARD_Z),)
    # ここ
    wTw2p = board_vertices[0]
    wPs = []
    bb_cord = homography_cam2pj()
    for i in range(4):

        # uv, _ = world2cam(p, modelview_matrix, projection_matrix, (display_width, display_height))
        uv = (bb_cord[i][0], bb_cord[i][1])
        cDc2p, cRw, cTc2w, _ = cam2world(uv, modelview_matrix, projection_matrix, (display_width, display_height))
        # projectionかけた後にzで割ったもの, 回転行列, 並進ベクトル

        wRc = cRw.T
        wDc2p = np.dot(wRc, cDc2p[0:3]) # 世界座標系での光線ベクトル （回転行列 × prjection後のuv座標）
        wTw2c = np.dot(wRc, -cTc2w)     # 世界座標系でのカメラの位置ベクトル （回転行列 × 負の並進ベクトル）
        wTc2p = wTw2p - wTw2c           # 世界座標系でのカメラから対象物までの光線ベクトル （世界座標系における壁の1点 - 世界座標系でのカメラの位置ベクトル）

        k = wTc2p[2] / wDc2p[2]         # 壁面までの光線長さ (世界座標系でのカメラから対象物までの光線ベクトル/世界座標系での光線ベクトル)

        wP = wTw2c + k * wDc2p # 光線ベクトルと壁面との交点 (カメラの位置ベクトルに光線の長さをかけた世界座標系での位置ベクトルを足したもの)
        wPs.append(wP)
        # print("aaa",wPs)
        #print(w_wallpoint)
        #objects.append(Sphere(radius=0.01, position=wP, diffuse=(0.2, 0.2, 0.2)))

    points = np.array(wPs)[:, 0:2]
    print("point\n",points)
    print("z", wTw2p[2] + 0.01)
    # hull = ConvexHull(points)
    # hull_points = hull.points[hull.vertices]
    # hull_points = np.insert(hull_points, 2, wTw2p[2] + 0.01 , axis=1)

    # pj1→cam 
    # rvec1 = np.loadtxt("rvec.txt")
    # inv_rvec1 = np.linalg.inv(rvec1)
    # tvec1 =  np.loadtxt("tvec.txt")
    # tvec1_T = np.transpose(tvec1)
    # tvec1_T = tvec1_T.reshape(3, 1)
    # print(tvec1)
    # print(tvec1_T)
    # # 回転並進の合成
    # Rt = inv_rvec1
    # Rt = np.hstack([Rt, -tvec1_T])
    # addH = np.array([0, 0, 0, 1])
    # Rt = np.vstack([Rt, addH])
    # print("Rt\n", Rt)

    # cam→pj2
    # rvec2 = np.loadtxt("rvec2.txt")
    # tvec2 = np.loadtxt("tvec2.txt")
    # tvec2_T = np.transpose(tvec2)
    # tvec2_T = tvec2_T.reshape(3, 1)
    # # 回転並進の合成
    # Rt2 = rvec2
    # Rt2 = np.hstack([Rt2, tvec2_T])
    # Rt2 = np.vstack([Rt2, addH])

    origin_point = []
    inv_modelview_matrix = np.linalg.inv(modelview_matrix)
    for i in range(4):
        shadow_cord = np.array([[points[i][0]], [points[i][1]], [wTw2p[2] + 0.01], [1.]])
        additional = np.dot(inv_modelview_matrix, shadow_cord)
        origin_point = np.append(origin_point, additional)
    origin_point = origin_point.reshape([4, 4])
    print("origin\n", origin_point)


    shadow_pj2 = []
    for i in range(4):
        origin_cord = np.array([[origin_point[i][0]], [origin_point[i][1]], [origin_point[i][2]], [1.]])
        additional = np.dot(modelview_matrix2, origin_cord)
        shadow_pj2 = np.append(shadow_pj2, additional)
    shadow_pj2 = shadow_pj2.reshape([4, 4])
    print("shadow_pj2\n", shadow_pj2)


    # 並進ベクトルの足し算
    # for i in range(4):
    #     kaiten_point = origin_point[i].T
    #     additional = kaiten_point + tvec1
    #     origin_point = np.append(origin_point, additional)
    # origin_point = origin_point.reshape([4, 3])
    # print("origin_point\n", origin_point)

"""
    homography_pjcm = np.loadtxt("homography_pjcm.txt")
    shadow_cam = []
    # 影をPJ1視点からカメラ視点への変換
    for i in range(4):
        shadow_cord = np.array([[points[i][0]], [points[i][1]], [wTw2p[2] + 0.01]])
        additional = np.dot(homography_pjcm, shadow_cord)
        # additional = additional / additional[2]
        shadow_cam = np.append(shadow_cam, additional)
    shadow_cam = shadow_cam.reshape([4, 3])
    print("shadow_cam\n", shadow_cam)


    # 影をカメラ視点からPJ2視点へ変換
    homography_pjcm2 = np.loadtxt("homography_pjcm2.txt")
    homography_cmpj2 = np.linalg.inv(homography_pjcm2)
    shadow_pj2 = []
    for i in range(4):
        shadow_cord = np.array([[shadow_cam[i][0]], [shadow_cam[i][1]], [shadow_cam[i][2]]])
        additional = np.dot(homography_cmpj2, shadow_cord)
        # additional = additional / additional[2]
        shadow_pj2 = np.append(shadow_pj2, additional)
    shadow_pj2 = shadow_pj2.reshape([4, 3])
    print("shadow_pj2\n", shadow_pj2)
"""


modelview_matrix = np.loadtxt("modelview_matrix.txt")
projection_matrix = np.loadtxt("projection_matrix.txt")
modelview_matrix2 = np.loadtxt("modelview_matrix2.txt")
plot_points_on_wall()