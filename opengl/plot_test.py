import cv2 
import numpy as np

np.set_printoptions(suppress=True)

BOARD_WIDTH  = 0.8  # chessboard の横幅 [m]
BOARD_HEIGHT = 0.45  # chessboard の縦幅 [m]
BOARD_X = 0.         # chessboard の3次元位置X座標 [m]（右手系）
BOARD_Y = 0.         # chessboard の3次元位置Y座標 [m]（右手系）
BOARD_Z = -3.0  
board_vertices = ((BOARD_X - BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT, BOARD_Z),
                (BOARD_X - BOARD_WIDTH / 2, BOARD_Y, BOARD_Z),
                (BOARD_X + BOARD_WIDTH / 2, BOARD_Y, BOARD_Z),
                (BOARD_X + BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT, BOARD_Z))

# スクリーンショットの適切な位置にボードの角の座標がプロットされるかの確認関数
# GLの三次元座標にmatrixをかけてuvにした値をスクショのボード位置にプロットしている
def world2cam():
  # スクリーン交点の座標
  # point3d = np.array([[1.065, 1.09, -3.0], [1.08, 1.113, -3.0], [1.094, 1.125, -3.0], [1.078, 1.102, -3.0]])

  point3d = np.array([[board_vertices[0][0]],[board_vertices[0][1]], [board_vertices[0][2]], [1.]])
  print('point3d', point3d)
  modelview_matrix = np.loadtxt("modelview_matrix.txt")
  projection_matrix = np.loadtxt("projection_matrix.txt")
  point2d = np.dot(modelview_matrix, point3d)
  point2d = np.dot(projection_matrix, point2d)
    # (w, h) = img_size
  (w, h) = 1536, 864
  u = (int)(w / 2 * (point2d[0] / point2d[2] + 1))
  v = (int)(h - h / 2 * (point2d[1] / point2d[2] + 1))
  print("u:{},v:{}".format(u, v))
  return u, v 

print("uv", world2cam())
u, v = 440, 64
# ホモグラフィの逆
homography = np.loadtxt("homography_pjcm.txt")
point = np.array([[u], [v], [1.]])
ans = np.dot(homography, point)
ans = ans / ans[2]
# ホモグラフィ
inv_homography = np.linalg.inv(np.loadtxt("homography_pjcm2.txt"))
ans = np.dot(inv_homography, ans)
ans = ans / ans[2]
print("ans\n", ans)
# u, v = 368, 234が出て欲しい

modelview2 = np.loadtxt("modelview_matrix2.txt")
projection2 = np.loadtxt("projection_matrix2.txt")


# 座標取り出し
img = cv2.imread("./pj1.png")
img = cv2.circle(img, (u, v), 15, (0, 255, 0), thickness=-1)
cv2.imwrite("./pj1_after.png", img)