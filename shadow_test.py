
import numpy as np
import cv2
from pyglet import window

np.set_printoptions(suppress=True)
"""
カメラで取得したBBの座標4点を一度プロジェクタ1視点の座標に変換し，物体との直線から壁との交点（影の位置）を算出し，算出した影の世界座標をpj1とpj2にuv変換するコード
"""


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



# 影の座標をそれぞれのプロジェクタのuvに落とす
def shadow2pj(points, modelview1, projection1, modelview2, projection2):
  print(points)
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


modelview_matrix = np.loadtxt("modelview_matrix.txt")
projection_matrix = np.loadtxt("projection_matrix.txt")
modelview_matrix2 = np.loadtxt("modelview_matrix2.txt")
projection_matrix2 = np.loadtxt("projection_matrix2.txt")
# points = plot_points_on_wall()
# ホモグラフィの推定をするために必要
BOARD_WIDTH = 0.8  # chessboard の横幅 [m]
BOARD_HEIGHT = 0.45  # chessboard の縦幅 [m]
BOARD_X = 0.  # chessboard の3次元位置X座標 [m]（右手系）
BOARD_Y = 0.  # chessboard の3次元位置Y座標 [m]（右手系）
BOARD_Z = -1.6
chess_cord = (
(BOARD_X - BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT, BOARD_Z), (BOARD_X - BOARD_WIDTH / 2, BOARD_Y, BOARD_Z),
(BOARD_X + BOARD_WIDTH / 2, BOARD_Y, BOARD_Z), (BOARD_X + BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT, BOARD_Z))

points = [[BOARD_X - BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT / 2, BOARD_Z],
          [BOARD_X - BOARD_WIDTH / 2, -0.225, BOARD_Z],
          [0, -0.225, BOARD_Z],
          [0, BOARD_Y + BOARD_HEIGHT / 2, BOARD_Z]]
points = np.loadtxt("shadow_cord.txt")
shadow2pj(points, modelview_matrix, projection_matrix, modelview_matrix2, projection_matrix2)

img = cv2.imread("pj1.png")
pj1_cord = np.loadtxt("pj1sh_cord.txt")
pts = [np.array(((int(pj1_cord[0][0]), int(pj1_cord[0][1])), (int(pj1_cord[1][0]), int(pj1_cord[1][1])), (int(pj1_cord[2][0]), int(pj1_cord[2][1])), (int(pj1_cord[3][0]), int(pj1_cord[3][1]))))]
color = (0, 255, 0)
cv2.polylines(img, pts, True, color, thickness=2)
# cv2.imwrite("pj1_after.png", img)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 外側塗り潰し
img = cv2.imread("pj2.png")
# stencil = np.zeros(img.shape).astype(img.dtype)
pj2_cord = np.loadtxt("pj2sh_cord.txt")
contours = [np.array(((int(pj2_cord[0][0]), int(pj2_cord[0][1])), (int(pj2_cord[1][0]), int(pj2_cord[1][1])), (int(pj2_cord[2][0]), int(pj2_cord[2][1])), (int(pj2_cord[3][0]), int(pj2_cord[3][1]))))]
# contours = [np.array(((int(pj2_cord[0][0]), int(pj2_cord[0][1])*-2), (int(pj2_cord[1][0]), int(pj2_cord[1][1])), (int(pj2_cord[2][0]), int(pj2_cord[2][1])), (int(pj2_cord[3][0]), int(pj2_cord[3][1])*-2)))]
color = (255, 255, 255)
# cv2.fillPoly(stencil, contours, color)
color = (0, 255, 0)
cv2.polylines(img, contours, True, color, thickness=2)
# result = cv2.bitwise_and(img, stencil)
# cv2.fillPoly(img, contours, color=[0, 0, 0])
# cv2.imwrite("pj2_after.png", img)
cv2.imshow("result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread("pj1.png")
pj1_cord = np.loadtxt("pjjjj_cord.txt")
pts = [np.array(((int(pj1_cord[0][0]), int(pj1_cord[0][1])), (int(pj1_cord[1][0]), int(pj1_cord[1][1])), (int(pj1_cord[2][0]), int(pj1_cord[2][1])), (int(pj1_cord[3][0]), int(pj1_cord[3][1]))))]
color = (0, 255, 0)
cv2.polylines(img, pts, True, color, thickness=2)
# cv2.imwrite("pj1_after.png", img)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()