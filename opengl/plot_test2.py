import numpy as np

from cord_calc import world2cam

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
  point2d = np.dot(modelview_matrix, point3d)
  point2d = np.dot(projection_matrix, point2d)
    # (w, h) = img_size
  (w, h) = 1920, 1080
  u = (int)(w / 2 * (point2d[0] / point2d[2] + 1))
  v = (int)(h - h / 2 * (point2d[1] / point2d[2] + 1))
  print("u:{},v:{}".format(u, v))
  return u, v

shadow_cord = np.array([[points[i][0]], [points[i][1]], [wTw2p[2] + 0.01], [1.]])
inv_modelview_matrix = np.loadtxt("modelview_matrix.txt")
inv_modelview_matrix = np.linalg.inv(inv_modelview_matrix)
modelview_matrix2 = np.linalg.inv("modelview_matrix2.txt")
additional = np.dot(inv_modelview_matrix, shadow_cord)
additional = np.dot(modelview_matrix2, additional)
u, v = world2cam()