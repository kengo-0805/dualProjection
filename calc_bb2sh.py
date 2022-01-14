
import numpy as np
import cv2

np.set_printoptions(suppress=True)
"""
カメラで取得したBBの座標4点を一度プロジェクタ1視点の座標に変換し，物体との直線から壁との交点（影の位置）を算出し，算出した影の世界座標をpj1とpj2にuv変換するコード
"""


def homography_cam2pj():
    pj_cord = []
    # # カメラ視点でのバウンディングボックスの座標の読み込み
    # bb_cord = np.loadtxt("bb_cord.txt")
    # if bb_cord is not None:
    #     print("bb_cord読み込みOK")
    #     print(bb_cord)

    # # カメラからプロジェクタへの射影変換行列
    # homography_pjcm = np.loadtxt("homography_pjcm.txt")
    # homography_cmpj = np.linalg.inv(homography_pjcm)
    # if homography_cmpj is not None:
    #     print("cm→pjのホモグラフィOK")
    #     # print(homography_cmpj)

    # ホモグラフィから回転並進への分解
    # color_intrinsics = np.loadtxt("color_intrinsics.txt")
    # _, R, t, _ = cv2.decomposeHomographyMat(homography_cmpj, color_intrinsics)
    # print(R)
    # print(t)
    # R = R[0]
    # t = t[0]
    
    # ステレオキャリブレーションで求めた
    R = np.loadtxt("R_cmpj.txt")
    t = np.loadtxt("t_cmpj.txt")
    t = t.reshape([3, 1])
    print(R)
    print(t)
    Rt = np.hstack((R, t))
    addH = np.array([0, 0, 0, 1])
    Rt = np.vstack((Rt, addH))
    print("Rt\n", Rt)

    # BBの3次元座標
    obj_xyz = np.loadtxt("obj_xyz.txt")
    for i in range(4):
      # obj_xyz[i] = obj_xyz[i] / obj_xyz[i][2]
      xyz = np.hstack((np.transpose(obj_xyz[i]), 1))

      print("xyz\n", xyz)
      additional = np.dot(Rt, xyz)
      additional = world2cam(additional[:3], modelview_matrix, projection_matrix, (1920, 1080))
      # print("add\n", additional)
      pj_cord = np.append(pj_cord, additional)
    pj_cord = pj_cord.reshape([4, 2])
    print("pj視点のBB\n", pj_cord)



    # # BBからPJ視点への変換
    # for i in range(4):
    #     bb_cord_array = np.array([[bb_cord[0][i]], [bb_cord[1][i]], [1.]])
    #     additional = np.dot(homography_cmpj, bb_cord_array)
    #     additional = additional / additional[2]
    #     pj_cord = np.append(pj_cord, additional)
    # pj_cord = pj_cord.reshape([4, 3])
    # np.savetxt("pjjjj_cord.txt", pj_cord)
    # print("pj視点のBB\n", pj_cord)

    return pj_cord


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

    return (u, v)


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
    # wTw2p = [-2.5, 0, -5]
    BOARD_X = 0.
    BOARD_Y = 0.
    BOARD_Z = -1.6
    BOARD_WIDTH = 0.8
    BOARD_HEIGHT = 0.45
    board_vertices = ((BOARD_X - BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT / 2, BOARD_Z),)
    # ここ
    wTw2p = board_vertices[0]
    wPs = []
    # bb_cordにpj視点に変換した座標を代入
    bbb_cord = homography_cam2pj()
    for i in range(4):

        # uv, _ = world2cam(p, modelview_matrix, projection_matrix, (display_width, display_height))
        uv = (bbb_cord[i][0], bbb_cord[i][1])
        cDc2p, cRw, cTc2w, _ = cam2world(uv, modelview_matrix, projection_matrix, (display_width, display_height))
        # projectionかけた後にzで割ったもの, 回転行列, 並進ベクトル

        wRc = cRw.T
        wDc2p = np.dot(wRc, cDc2p[0:3]) # 世界座標系での光線ベクトル （回転行列 × prjection後のuv座標）
        wTw2c = np.dot(wRc, -cTc2w)     # 世界座標系でのカメラの位置ベクトル （回転行列 × 負の並進ベクトル）
        # print("R", wRc)
        # print("-t", -cTc2w)
        wTc2p = wTw2p - wTw2c           # 世界座標系でのカメラから対象物までの光線ベクトル （世界座標系における壁の1点 - 世界座標系でのカメラの位置ベクトル）

        k = wTc2p[2] / wDc2p[2]         # 壁面までの光線長さ (世界座標系でのカメラから対象物までの光線ベクトル/世界座標系での光線ベクトル)

        wP = wTw2c + k * wDc2p # 光線ベクトルと壁面との交点 (カメラの位置ベクトルに光線の長さをかけた世界座標系での位置ベクトルを足したもの)
        wPs.append(wP)
        # print("aaa",wPs)
        #print(w_wallpoint)
        #objects.append(Sphere(radius=0.01, position=wP, diffuse=(0.2, 0.2, 0.2)))

    points = np.array(wPs)[:, 0:3]
    print("shadow_point\n",points)
    print("z", wTw2p[2] + 0.01)
    np.savetxt("shadow_cord.txt", points) # いらんかも
    return points
    # hull = ConvexHull(points)
    # hull_points = hull.points[hull.vertices]
    # hull_points = np.insert(hull_points, 2, wTw2p[2] + 0.01 , axis=1)


# 影の座標をそれぞれのプロジェクタのuvに落とす
def shadow2pj(points, modelview1, projection1, modelview2, projection2):
  pj1_cord = []
  pj2_cord = []
  # ホモグラフィの推定をするために必要
  BOARD_WIDTH  = 0.8  # chessboard の横幅 [m]
  BOARD_HEIGHT = 0.45  # chessboard の縦幅 [m]
  BOARD_X = 0.         # chessboard の3次元位置X座標 [m]（右手系）
  BOARD_Y = 0.         # chessboard の3次元位置Y座標 [m]（右手系）
  BOARD_Z = -1.6 
  chess_cord =  ((BOARD_X - BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT, BOARD_Z),(BOARD_X - BOARD_WIDTH / 2, BOARD_Y, BOARD_Z),                (BOARD_X + BOARD_WIDTH / 2, BOARD_Y, BOARD_Z), (BOARD_X + BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT, BOARD_Z))
# チェスボードの角の座標をuvにそれぞれ変換する
  for i in range(4): 
    # w, h = window.get_size()
    (w, h) = 1920, 1080 
    additional1 = world2cam(chess_cord[i], modelview1, projection1, (w, h))
    additional2 = world2cam(chess_cord[i], modelview2, projection2, (w, h))
    pj1_cord = np.append(pj1_cord, additional1) 
    pj2_cord = np.append(pj2_cord, additional2)
  pj1_cord = pj1_cord.reshape([4, 2])
  pj2_cord = pj2_cord.reshape([4, 2])
  print("pj1chess\n", pj1_cord)
  np.savetxt("pj1_cord.txt", pj1_cord)
  print("pj2chess\n", pj2_cord)
  np.savetxt("pj2_cord.txt", pj2_cord)

# 影の座標をそれぞれの視点に変換する
  pj1sh_cord = []
  pj2sh_cord = []
  for i in range(4): 
    # w, h = window.get_size()
    (w, h) = 1920, 1080
    additional1 = world2cam(points[i], modelview1, projection1, (w, h))
    additional2 = world2cam(points[i], modelview2, projection2, (w, h))
    pj1sh_cord = np.append(pj1sh_cord, additional1) 
    pj2sh_cord = np.append(pj2sh_cord, additional2)
  pj1sh_cord = pj1sh_cord.reshape([4, 2])
  pj2sh_cord = pj2sh_cord.reshape([4, 2])
  print("pj1sh\n", pj1sh_cord)
  np.savetxt("pj1sh_cord.txt", pj1sh_cord)
  print("pj2sh\n", pj2sh_cord)
  np.savetxt("pj2sh_cord.txt", pj2sh_cord)

  # # test
  # print(pj2_cord[0][0])
  # img = cv2.imread("./pj2.png")
  # for i in range(4):
  #   img = cv2.circle(img, (int(pj2_cord[i][0]), int(pj2_cord[i][1])), 15, (0, 255, 0), thickness=-1)
  # cv2.imwrite("./pj2_after.png", img)

modelview_matrix = np.loadtxt("modelview_matrix.txt")
projection_matrix = np.loadtxt("projection_matrix.txt")
modelview_matrix2 = np.loadtxt("modelview_matrix2.txt")
projection_matrix2 = np.loadtxt("projection_matrix2.txt")
points = plot_points_on_wall()
# BOARD_WIDTH = 0.8  # chessboard 1の横幅 [m]
# BOARD_HEIGHT = 0.45  # chessboard の縦幅 [m]
# BOARD_X = 0.  # chessboard の3次元位置X座標 [m]（右手系）
# BOARD_Y = 0.  # chessboard の3次元位置Y座標 [m]（右手系）
# BOARD_Z = -1.6
# points = [[BOARD_X - BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT / 2, BOARD_Z],
#           [BOARD_X - BOARD_WIDTH / 2, -0.225, BOARD_Z],
#           [0, -0.225, BOARD_Z],
#           [0, BOARD_Y + BOARD_HEIGHT / 2, BOARD_Z]]
shadow2pj(points, modelview_matrix, projection_matrix, modelview_matrix2, projection_matrix2)