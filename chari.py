#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np

global height
global width

square_size = 2.2      # 正方形の1辺のサイズ[cm]
pattern_size = (8, 13)  # 交差ポイントの数

reference_img = 10 # 参照画像の枚数

pattern_points = np.zeros( (np.prod(pattern_size), 3), np.float32 ) #チェスボード（X,Y,Z）座標の指定 (Z=0)
pattern_points[:,:2] = np.indices(pattern_size).T.reshape(-1, 2)
pattern_points *= square_size
objpoints = []
imgpoints = []

capture = cv2.VideoCapture(0)

while len(objpoints) < reference_img:
# 画像の取得
    ret, img = capture.read()
    height = img.shape[0]
    width = img.shape[1]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    cv2.imshow('image', img)
    # 毎回判定するから 200 ms 待つ．遅延するのはココ
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

print("calculating camera parameter...")
# 内部パラメータを計算
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None) # 再投影誤差，A，歪み，R，t

# 計算結果を保存
np.save("mtx", mtx) # カメラ行列
np.save("dist", dist.ravel()) # 歪みパラメータ
# # 計算結果を表示
# print("RMS = ", ret)
# print("mtx = \n", mtx)
# print("dist = ", dist.ravel())
print('rvec', rvecs)
# 1番目の回転しか使っていないので怪しい
R, _ = cv2.Rodrigues(rvecs[0])
R = np.delete(R, 2, 1)
# print("R",R)
# print("t", tvecs[0])
# 1番目の並進ベクトルしかかけてないので怪しい
Rt = np.hstack((R, tvecs[0]))
# print("Rt",Rt)
# H_srcm = np.dot(mtx, Rt)
# inv_Hsrcm = np.linalg.inv(H_srcm)

H_pjcm = np.loadtxt("homography_pjcm.txt")
inv_Hpjcm = np.linalg.inv(H_pjcm)
# 使うのか?
# Hpjsr = np.dot(inv_Hsrcm, H_pjcm)
# print("Hpjsr", Hpjsr)

imgpoints2 = []

for i in range(10):
  for j in range(len(imgpoints[0])):
    doumae = np.array([[imgpoints[i][j][0]], [imgpoints[i][j][1]], [1]])
    addition = np.dot(inv_Hpjcm, doumae)
    addition = addition / addition[2]
    addition = np.delete(addition, 2, 0)
    imgpoints2 = np.append(imgpoints2, addition)
imgpoints2 = imgpoints2.reshape([10, len(objpoints[0]), 2])
# imgpoints2 = imgpoints2.tolist()

_objpoints = []
_imgpoints = []
_imgpoints2 = []
for o, i1, i2 in zip(objpoints, imgpoints, imgpoints2):
  _objpoints.append(o)
  _imgpoints.append(i1)
  _imgpoints2.append(i2.astype(np.float32))
  
# print(len(objpoints))
# print(len(imgpoints))
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
# print("img2shape", imgpoints2.shape)
ret, l_mat_new, l_dist_new, r_mtx_new, r_dist_new, R_cmpj, T_cmpj, E, F = cv2.stereoCalibrate(_objpoints, _imgpoints, _imgpoints2, mtx, dist, mtx, 0, (width, height))
np.savetxt("R_cmpj.txt", R_cmpj)
np.savetxt("t_cmpj.txt", T_cmpj)
print("RRR", R_cmpj)
print("ttt", T_cmpj)