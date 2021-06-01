import os

import cv2
from PIL import Image
import pyglet
import pyglet.gl as gl
import pyrealsense2 as rs

import math
import numpy as np
import time

# -------------------------------
# ArUcoの読み取り箇所
# -------------------------------

aruco = cv2.aruco
# マーカーの辞書選択
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
# VideoCapture オブジェクトを取得します
capture = cv2.VideoCapture(0)
# マーカーのサイズ
marker_length = 0.056 # [m]

camera_matrix = np.load("mtx.npy")
'''np.array(([956.78928232,   0.,         645.67414608,],
                        [  0.,         955.45250146, 367.70419916,],
                        [  0.,           0.,           1.        ]))'''
distortion_coeff = np.load("dist.npy") #np.array(([0,0,0,0,0]))


while True:
    start = time.time()
    ret, img = capture.read()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)
    # 可視化
    aruco.drawDetectedMarkers(img, corners, ids) 
    # resize the window
    windowsize = (800, 600)
    img = cv2.resize(img, windowsize)
    

    cv2.imshow('title',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(corners) > 0:
        # マーカーごとに処理
        for i, corner in enumerate(corners):
            # rvec -> rotation vector, tvec -> translation vector
            rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, distortion_coeff)
            # 回転ベクトルからrodoriguesへ変換
            rvec_matrix = cv2.Rodrigues(rvec)
            rvec_matrix = rvec_matrix[0] # rodoriguesから抜き出し
            kado = corners[0]
            print("id ={}".format(ids[i]))
            print(tvec)
            print("corners={}".format(corners[0]))
            print("corners={}".format(corners[0][0]))
            print("corners={}".format(corners[0][0][0]))
            
            if ids[i] == 0:
                print("0を認識")
                f = open("id0.txt","w")
                f.write("{}".format(tvec))
                f.close()
                '''
                lef_upX = kado[0][0][0]
                lef_upY = kado[0][0][1]
                # lef_upZ = tvec[0][2]
                print("x = {}".format(lef_upX))
                print("y = {}".format(lef_upY))
                # print("z = {}".format(lef_upZ))
                '''
            
            # if ids[i] == 1:
            #     print("1を認識")
            #     n = open("id1.txt","w")
            #     n.write("{}".format(tvec))
            #     n.close()
            #     '''
            #     rig_upX = kado[1][0][0]
            #     rig_upY = kado[1][0][1]
            #     print("x = {}".format(rig_upX))
            #     print("y = {}".format(rig_upY))
            #     '''
            # if ids[i] == 2:
            #     print("2を認識")
            #     n = open("id2.txt","w")
            #     n.write("{}".format(tvec))
            #     n.close()
            
            # if ids[i] == 3:
            #     print("3を認識")
            #     n = open("id3.txt","w")
            #     n.write("{}".format(tvec))
            #     n.close()
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")
capture.release()
cv2.destroyAllWindows()