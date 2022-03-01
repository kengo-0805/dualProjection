import numpy as np
import cv2
from numpy.ma.core import dot

from pyglet.window.key import P

# [see] CG 内の3次元座標を画像へ投影
def world2cam(point3d, mv_matrix, pj_matrix, img_size):
    """
    :param point3d: 世界座標系での三次元座標
    :param mv_matrix:
    :param pj_matrix:
    :param img_size:
    :return:
    """
    # _point3d = np.hstack((point3d, 1))
    _point3d = np.dot(mv_matrix, point3d) # カメラ座標系の3次元点
    _point2d = np.dot(pj_matrix, _point3d) # 正規化デバイス座標系での標準視体積内の表現

    # 同次座標の不定性の解消（s_1 = _point2d[2]）
    (w, h) = img_size
    u = (int)(w / 2 * (_point2d[0] / _point2d[2] + 1))
    v = (int)(h - h / 2 * (_point2d[1] / _point2d[2] + 1))

    return (u, v)

# 座標を入力→Rtで変換→world2camで落とす→画像にプロット
cord1 = np.array([[-0.3561234176158905], [0.12291482836008072], [1.6230000257492065]])
cord2 = np.array([[-0.35175296664237976], [0.46768710017204285], [1.6350000858306885]])
cord3 = np.array([[0.3528285622596741], [0.4821659028530121], [1.5910000801086426]])
cord4 = np.array([[0.34528470039367676], [0.13227175176143646], [1.6410000324249268]])
# cord1 = np.array([[-0.12835434079170227], [0.3086690902709961], [1.0240000486373901]])
# cord2 = np.array([ [-0.11922897398471832], [0.3709619343280792], [1.0170000791549683]])
# cord3 = np.array([[-0.02369171939790249], [0.36484494805336], [1.0230000019073486]])
# cord4 = np.array([ [-0.015500455163419247], [0.3088052272796631], [1.03000009059906]])
cord1 = np.array([[0.11204022914171219], [0.3021131157875061], [1.0840001106262207]])
cord2 = np.array([[0.11243375390768051], [0.3576172888278961], [1.071000099182129]])
cord3 = np.array([[0.35585594177246094], [0.5370913743972778], [1.593000054359436]])
cord4 = np.array([[0.34089311957359314], [0.4575727581977844], [1.5600000619888306]])

R = np.array([[ 0.97469931,  0.03872942,  0.22013926],
 [ 0.00090655,  0.88339792, -0.18145218],
 [-0.22351203,  0.17728102,  0.85844343]])

theta1 = 15 # 上
theta2 = 4 # 左
theta3 = 0 # 左回転
c1 = np.cos(theta1 * np.pi / 180)
s1 = np.sin(theta1 * np.pi / 180)
c2 = np.cos(theta2 * np.pi / 180)
s2 = np.sin(theta2 * np.pi / 180)
c3 = np.cos(theta3 * np.pi / 180)
s3 = np.sin(theta3 * np.pi / 180)
R = np.array([[c2*c3, -c2*s3, s2],
                [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
# R = np.linalg.inv(R)
t = np.array([[-0.02994892], # 左
 [ 0.031], # 下
 [ -0.02917737]]) # 縮小？
t = -t.reshape([3, 1])

modelview_matrix = np.loadtxt("modelview_matrix.txt")
projection_matrix = np.loadtxt("projection_matrix.txt")

# Rtの合成
Rt = np.hstack((R, t))
addH = np.array([0, 0, 0, 1])
Rt = np.vstack((Rt, addH))
print("Rt\n", Rt)

# プロジェクタ視点への変換
# 左上
cv2gl = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
xyz = np.dot(cv2gl, cord1)
xyz = np.vstack((xyz, 1))
xyz = np.transpose(xyz)
xyz = np.transpose(xyz)
pj_xyz = np.dot(Rt, xyz)
print("pj_xyz\n", pj_xyz)
# 右下
xyz = np.dot(cv2gl, cord2)
xyz = np.vstack((xyz, 1))
xyz = np.transpose(xyz)
xyz = np.transpose(xyz)
pj_xyz2 = np.dot(Rt, xyz)
print("pj_xyz\n", pj_xyz2)
# 左下
xyz = np.dot(cv2gl, cord3)
xyz = np.vstack((xyz, 1))
xyz = np.transpose(xyz)
xyz = np.transpose(xyz)
pj_xyz3 = np.dot(Rt, xyz)
print("pj_xyz\n", pj_xyz3)
# 左上
xyz = np.dot(cv2gl, cord4)
xyz = np.vstack((xyz, 1))
xyz = np.transpose(xyz)
xyz = np.transpose(xyz)
pj_xyz4 = np.dot(Rt, xyz)
print("pj_xyz\n", pj_xyz4)

# 3次元座標をuvに変換
img = cv2.imread("./pj1.png")
h, w = img.shape[:2]
u0, v0 = world2cam(pj_xyz, modelview_matrix, projection_matrix, (w, h))
u1, v1 = world2cam(pj_xyz2, modelview_matrix, projection_matrix, (w, h))
u2, v2 = world2cam(pj_xyz3, modelview_matrix, projection_matrix, (w, h))
u3, v3 = world2cam(pj_xyz4, modelview_matrix, projection_matrix, (w, h))
print("左上", u0, v0)
print("右下", u1, v1)

# プロット
dot_img = cv2.circle(img, (u0, v0), 15, (0, 0, 255), thickness=-1)
dot_img = cv2.circle(dot_img, (u1, v1), 15, (0, 255, 0), thickness=-1)
dot_img = cv2.circle(dot_img, (u2, v2), 15, (255, 0, 0), thickness=-1)
dot_img = cv2.circle(dot_img, (u3, v3), 15, (255, 255, 0), thickness=-1)
cv2.imwrite("dot_img.png", dot_img)
cv2.imshow("dot_img", dot_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
