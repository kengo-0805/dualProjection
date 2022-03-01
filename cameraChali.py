#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


square_size = 0.05      # 正方形の1辺のサイズ[cm]
pattern_size = (8, 15)  # 交差ポイントの数

reference_img = 5 # 参照画像の枚数

threshold = 100

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
    img = cv2.imread("./align/align_img_c{}.png".format(i))
    height = img.shape[0]
    width = img.shape[1]
    print(width,height)

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
    cv2.imshow('image', img)
    # 毎回判定するから 200 ms 待つ．遅延するのはココ
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break
    # time.sleep(2)
print("calculating camera parameter...")
# 内部パラメータを計算
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 計算結果を保存
# np.save("mtx", mtx) # カメラ行列
np.savetxt("mtx.txt", mtx)
# np.save("dist", dist.ravel()) # 歪みパラメータ
np.savetxt("dist.txt", dist)
# 計算結果を表示
print("RMS = ", ret)
print("mtx = \n", mtx)
print("dist = ", dist.ravel())

img = cv2.imread("./align_img_.png")
h,  w = img.shape[:2]
mtx = np.array([[616.517, 0, 313.278], [0, 615.858, 246.359], [0, 0, 1]])
dist = np.array([0, 0, 0, 0, 0])

newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult_undistort.png',dst)

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite('calibresult_remap.png',dst)