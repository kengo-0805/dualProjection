#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import time

global height
global width

square_size = 0.05      # 正方形の1辺のサイズ[cm]
pattern_size = (8, 15)  # 交差ポイントの数

reference_img = 5 # 参照画像の枚数

threshold = 120

pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
objpoints = []
imgpoints = []

# capture = cv2.VideoCapture(1)

# while len(objpoints) < reference_img:
for i in range(reference_img):
# 画像の取得
    # ret, img = capture.read()
    # 解像度640x480
    img = cv2.imread("./align/align_img_c{}.png".format(i))
    # img = img.astype(np.uint8)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    height = img.shape[0]
    width = img.shape[1]


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, gray = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    # チェスボードのコーナーを検出
    ret, corner = cv2.findChessboardCorners(gray, pattern_size)
    # コーナーがあれば
    if ret == True:
        print("detected coner!")
        print(str(len(objpoints)+1) + "/" + str(reference_img))
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
        cv2.cornerSubPix(gray, corner, (5,5), (-1,-1), term)
        imgpoints.append(corner.reshape(-1, 2))   #appendメソッド：リストの最後に因数のオブジェクトを追加
        objpoints.append(pattern_points)
    print(ret)
    # print("len", len(imgpoints))
    cv2.imshow('image', gray)
    # 毎回判定するから 200 ms 待つ．遅延するのはココ
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
    # time.sleep(2)
print("calculating camera parameter...")
# 内部パラメータを計算
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) # 再投影誤差，A，歪み，R，t

# 計算結果を保存
# np.save("mtx", mtx) # カメラ行列
# np.save("dist", dist.ravel()) # 歪みパラメータ
# print("rms1", ret)
# print("mtx1", mtx)

# print('rvec', rvecs)
# 1番目の回転しか使っていないので怪しい
# R, _ = cv2.Rodrigues(rvecs[0])
# R = np.delete(R, 2, 1)
# print("R",R)
# print("t", tvecs)
# 1番目の並進ベクトルしかかけてないので怪しい
# Rt = np.hstack((R, tvecs[0]))
# print("Rt",Rt)
# H_srcm = np.dot(mtx, Rt)
# inv_Hsrcm = np.linalg.inv(H_srcm)

H_pjcm = np.loadtxt("homography_pjcm.txt")
inv_Hpjcm = np.linalg.inv(H_pjcm)
# 使うのか?
# Hpjsr = np.dot(inv_Hsrcm, H_pjcm)
# print("Hpjsr", Hpjsr)

# ホモグラフィの読み込み
H_pjcm_0 = np.loadtxt("Hpjcm_0.txt")
inv_Hpjcm_0 = np.linalg.inv(H_pjcm_0)
H_pjcm_1 = np.loadtxt("Hpjcm_1.txt")
inv_Hpjcm_1 = np.linalg.inv(H_pjcm_1)
H_pjcm_2 = np.loadtxt("Hpjcm_2.txt")
inv_Hpjcm_2 = np.linalg.inv(H_pjcm_2)
H_pjcm_3 = np.loadtxt("Hpjcm_3.txt")
inv_Hpjcm_3 = np.linalg.inv(H_pjcm_3)
H_pjcm_4 = np.loadtxt("Hpjcm_4.txt")
inv_Hpjcm_4 = np.linalg.inv(H_pjcm_4)

imgpoints2 = []
# 5回分ホモグラフィをかける
for j in range(len(imgpoints[0])):
  doumae = np.array([[imgpoints[0][j][0]], [imgpoints[0][j][1]], [1]])
  addition = np.dot(inv_Hpjcm_0, doumae)
  addition = addition / addition[2]
  addition = np.delete(addition, 2, 0)
  imgpoints2 = np.append(imgpoints2, addition)
for j in range(len(imgpoints[0])):
  doumae = np.array([[imgpoints[1][j][0]], [imgpoints[1][j][1]], [1]])
  addition = np.dot(inv_Hpjcm_1, doumae)
  addition = addition / addition[2]
  addition = np.delete(addition, 2, 0)
  imgpoints2 = np.append(imgpoints2, addition)
for j in range(len(imgpoints[0])):
  doumae = np.array([[imgpoints[2][j][0]], [imgpoints[2][j][1]], [1]])
  addition = np.dot(inv_Hpjcm_2, doumae)
  addition = addition / addition[2]
  addition = np.delete(addition, 2, 0)
  imgpoints2 = np.append(imgpoints2, addition)
for j in range(len(imgpoints[0])):
  doumae = np.array([[imgpoints[3][j][0]], [imgpoints[3][j][1]], [1]])
  addition = np.dot(inv_Hpjcm_3, doumae)
  addition = addition / addition[2]
  addition = np.delete(addition, 2, 0)
  imgpoints2 = np.append(imgpoints2, addition)
for j in range(len(imgpoints[0])):
  doumae = np.array([[imgpoints[4][j][0]], [imgpoints[4][j][1]], [1]])
  addition = np.dot(inv_Hpjcm_4, doumae)
  addition = addition / addition[2]
  addition = np.delete(addition, 2, 0)
  imgpoints2 = np.append(imgpoints2, addition)
imgpoints2 = imgpoints2.reshape([5, len(objpoints[0]), 2])


test_img = cv2.imread("pj1.png")
print("img2Point0", imgpoints2[4][0])
test_img = cv2.circle(test_img, (int(imgpoints2[4][60][0]), int(imgpoints2[4][60][1])), 15, (255, 255, 0), thickness=-1)
cv2.imwrite("test=img.png", test_img)
# for i in range(5):
#   for j in range(len(imgpoints[0])):
#     doumae = np.array([[imgpoints[i][j][0]], [imgpoints[i][j][1]], [1]])
#     addition = np.dot(inv_Hpjcm, doumae)
#     addition = addition / addition[2]
#     addition = np.delete(addition, 2, 0)
#     imgpoints2 = np.append(imgpoints2, addition)
# imgpoints2 = imgpoints2.reshape([reference_img, len(objpoints[0]), 2])



# # カメラ視点でのチェスボード座標にカメラからプロジェクタへのHをかけてプロジェクタ視点（imgpoint2）にする
# for i in range(reference_img):
#   for j in range(len(imgpoints[0])):
#     doumae = np.array([[imgpoints[i][j][0]], [imgpoints[i][j][1]], [1]])
#     addition = np.dot(inv_Hpjcm, doumae)
#     addition = addition / addition[2]
#     addition = np.delete(addition, 2, 0)
#     imgpoints2 = np.append(imgpoints2, addition)
# imgpoints2 = imgpoints2.reshape([reference_img, len(objpoints[0]), 2])
# imgpoints2 = imgpoints2.tolist()

# 型を揃えて入れ直し
_objpoints = []
_imgpoints = []
_imgpoints2 = []
for o, i1, i2 in zip(objpoints, imgpoints, imgpoints2):
  _objpoints.append(o)
  _imgpoints.append(i1)
  _imgpoints2.append(i2.astype(np.float32))
  
print(len(objpoints))
print(len(imgpoints))
# print(len(imgpoints2))

# for o, i1, i2 in zip(_objpoints, _imgpoints, _imgpoints2):
  # print(o.dtype)
  # print(i1.dtype)
  # print(i2.dtype)

# print("obj,img1,img2",objpoints.shape,imgpoints.shape, imgpoints2.shape)
# print(imgpoints2)
# print(type(objpoints))
# print(type(imgpoints))
# print(type(imgpoints2))
# print("objshape", objpoints.shape)
# print("img1shape", imgpoints.shape)
print("img2shape", imgpoints2.shape)
# print(imgpoints2[9].shape)

# print(len(_objpoints[9]))
# print(len(_imgpoints[9]))
# print(len(_imgpoints2[9]))
# print(_objpoints[9])
# print(_imgpoints[9].dtype)
# print(_objpoints[9].dtype)
# print(_imgpoints2[9].dtype)

# 鹿児島大？の手法により取得した内部パラメータ
P_mat = np.array([[ 2831.844,    0.000,  430.056],
                 [  0.000, 2897.241,  470.453],
                 [  0.000,    0.000,    1.000]])

# 内部パラメータの検証-------------------------
img = cv2.imread("./pj1.png")
h,  w = img.shape[:2]
# mtx = np.array([[616.517, 0, 313.278], [0, 615.858, 246.359], [0, 0, 1]])
# dist = np.array([0, 0, 0, 0, 0])

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(P_mat,0,(w,h),1,(w,h))
# undistort
dst = cv2.undistort(img, P_mat, 0, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult_undistort.png',dst)

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(P_mat, 0, None, newcameramtx, (w,h), 5)
dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult_remap.png',dst)
# -------------------------

# realsenseのカメラキャリブレーションによる内部パラメータと歪み
mtx = np.loadtxt("mtx.txt")
dist = np.loadtxt("dist.txt")
print(dist)
dist = 0
ret, l_mat_new, l_dist_new, r_mtx_new, r_dist_new, R_cmpj, T_cmpj, E, F = cv2.stereoCalibrate(_objpoints, _imgpoints, _imgpoints2, mtx, dist, P_mat, 0, (width, height))
np.savetxt("R_cmpj.txt", R_cmpj)
np.savetxt("t_cmpj.txt", T_cmpj)
print("RRR", R_cmpj)
rvec, _ = cv2.Rodrigues(R_cmpj)
print("rvec", rvec)
print("ttt", T_cmpj)
print("rms2", ret)
print("mtx2", l_mat_new)
# depth_intrinsics
# [ 640x480  p[318.228 239.711]  f[384.239 384.239]  Brown Conrady [0 0 0 0 0] ]
#
# color_intrinsics
# [ 640x480  p[323.218 242.496]  f[621.011 621.158]  Inverse Brown Conrady [0 0 0 0 0] ]

