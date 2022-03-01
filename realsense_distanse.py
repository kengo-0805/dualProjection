# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
from PIL import Image
import math
# import open3d as o3d

WIDTH = 640
HEIGHT = 480
THRESHOLD = 0.9 # これより近い距離の画素を無視する
SCREEN = 1.4

TARGET = 0.1 # 対象物のスクリーンからの距離
# color format
# データ形式の話
color_stream, color_format = rs.stream.color, rs.format.bgr8
depth_stream, depth_format = rs.stream.depth, rs.format.z16

# ストリーミング初期化
# RealSenseからデータを受信するための準備
# config.enable_streamでRGB，Dの解像度とデータ形式，フレームレートを指定している
config = rs.config()
config.enable_stream(depth_stream, WIDTH, HEIGHT, depth_format, 30)
config.enable_stream(color_stream, WIDTH, HEIGHT, color_format, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

# # 内部パラメータaa
depth_intrinsics = rs.video_stream_profile(profile.get_stream(rs.stream.depth)).get_intrinsics()
# print("color_intrinsics")
# print(color_intrinsics)
#color_intrinsics = np.array([[621.011, 0., 323.218], [0., 621.158, 242.496], [0., 0., 1.0]])
#np.savetxt("color_intrinsics.txt", color_intrinsics)
#depth_intrinsics = np.array([[384.239, 0., 318.228], [0., 384.239, 239.711], [0., 0., 1.0]])
# 1距離[m] = depth * depth_scale
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
# clipping_distance_in_meters = 0.4 # 40cm以内を検出
# depth値にする
# clipping_distance = clipping_distance_in_meters / depth_scale

# Alignオブジェクト生成
# RGBとDの画角の違いによるズレを修正している
align_to = rs.stream.color
align = rs.align(align_to)
# 検出とプリントするための閾値
# threshold = (WIDTH * HEIGHT * 3) * 0.9
max_dist = THRESHOLD / depth_scale

# # Realsense の内部パラメータの確認と保存
# def show_and_save_intrinsic(intrinsic, saveFlag = False):
#     # 内部パラメータの表示
#     print("====================================")
#     print("intrinsic parameters")
#     print("====================================")
#     print("\tintrinsic.width", intrinsic.width)
#     print("\tintrinsic.height", intrinsic.height)
#     print("\tintrinsic.fx", intrinsic.fx)
#     print("\tintrinsic.fy", intrinsic.fy)
#     print("\tintrinsic.ppx", intrinsic.ppx)
#     print("\tintrinsic.ppy", intrinsic.ppy)

#     if saveFlag:
#         np.savez('intrinsic.npz', width = intrinsic.width, height = intrinsic.height, fx = intrinsic.fx, fy = intrinsic.fy, ppx = intrinsic.ppx, ppy = intrinsic.ppy)


try:
    while True:
        # フレーム待ち（color&depth）
        # フレーム取得
        frames = pipeline.wait_for_frames()
        # フレームの画角差を修正
        aligned_frames = align.process(frames)
        # フレームの切り分け
        # 多分これに射影変換行列をかけたら視点の変更ができる
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue



        # dist = depth_frame.get_distance(x, y)

        # RGB画像のフレームから画素値をnumpy配列に変換
        # これで普通のRGB画像になる
        color_image = np.asanyarray(color_frame.get_data())
        cv2.imwrite("align_img_.png", color_image)
        

        # D画像のフレームから画素値をnumpy配列に変換
        depth_image = np.asanyarray(depth_frame.get_data()) # 深度の画素値が入っている

        # スクリーンの距離を取得するために3点取得



        # 指定距離以下を無視した深度画像の生成
        # 最大値より遠いものには情報を付与する的な？
        depth_filterd_image = (depth_image > max_dist) * depth_image
        depth_gray_filterd_image = (depth_filterd_image * 255. /max_dist).reshape((HEIGHT, WIDTH)).astype(np.uint8)

        # 指定距離以下を無視したRGB画像の生成
        color_filterd_image = (depth_filterd_image.reshape((HEIGHT, WIDTH, 1)) > 0) * color_image

        # coverage = [0]*64
        for y in range(HEIGHT):
            for x in range(WIDTH):
                dist = depth_frame.get_distance(x, y)
                if THRESHOLD < dist and dist < SCREEN - TARGET + 0.05: # 閾値以上スクリーン以下であれば
                # リストにその座標を格納するかその画素を消してしまうか
                    color_filterd_image[y, x] = [0, 255, 0]
                #     coverage[x//10] += 1

            # if y%20 is 19:
            #     line = ""
            #     for c in coverage:
            #         line += " .:nhBXWW"[c//25]
            #     coverage = [0]*64
            #     print(line)


        # #--------------------------------
        # # バウンディングボックス部分
        # #--------------------------------
        xyz = []
        # 形式変換
        # hsv = cv2.cvtColor(color_filterd_image, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv image",hsv)
        binary = cv2.inRange(color_filterd_image, (0, 254, 0), (0, 255, 0))
        cv2.imshow("niti",binary)
        # 輪郭抽出
        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        # 面積が一定以上の輪郭のみ残す。
        area_thresh = 1000
        contours = list(filter(lambda x: cv2.contourArea(x) > area_thresh, contours)) #xを与えてそのエリアが閾値より大きければリスト
        # # 輪郭を矩形で囲む。
        for cnt in contours:
            # 輪郭に外接する長方形を取得する。
            x, y, width, height = cv2.boundingRect(cnt)
            # 描画する。
            cv2.rectangle(color_image, (x - 10, y - 10), (x + width + 5, y + height), color=(0, 255, 0), thickness=2)
            # cv2.polylines(color_image, cnt, True, (255, 0, 0), 5)
            print("左上:{},{}".format(x, y))
            print("右下:{},{}".format(x + width, y + height))

            # # 左上
            # f = open("box_leftupcord.text","w")
            # f.write("{},{}".format(x,y))
            # f.close()
            # # 左下
            # f = open("box_leftdowncord.text","w")
            # f.write("{},{}".format(x,y+height))
            # f.close()
            # # 右下
            # f = open("box_rightupcord.text","w")
            # f.write("{},{}".format(x+width,y+height))
            # f.close()
            # # 右上
            # f = open("box_rightupcord.text","w")
            # f.write("{},{}".format(x+width,y))
            # f.close()


            # リスト
            x = float(x)
            y = float(y)
            width = float(width)
            height = float(height)
            bb_cord = [[x, x, x + width, x + width], [y, y + height, y + height, y]]
            # f = open("bb_cord.txt", "w")
            # f.write("{}".format(bb_cord))
            # f.close()
            np.savetxt("bb_cord.txt", bb_cord)
            print(bb_cord)

# ボックスのuv→xyz変換
            z = rs.depth_frame.get_distance(depth_frame, int(x + width/2), int(y + height/2))
        for i in range(4):
            print(i)
            xyz_cord = rs.rs2_deproject_pixel_to_point(depth_intrinsics, (float(bb_cord[0][i]), float(bb_cord[1][i])), z)
            # cvRgl = np.array([[1, 0, 0],[0, -1, 0], [0, 0, -1]])
            # xyz_cord = np.dot(cvRgl, xyz_cord)
            xyz = np.append(xyz, [xyz_cord])
            print("object pos = {}".format(xyz))
        xyz = xyz.reshape([4, 3])
        tate = math.sqrt((xyz[0][0]-xyz[1][0])*(xyz[0][0]-xyz[1][0]) + (xyz[0][1]-xyz[1][1])*(xyz[0][1]-xyz[1][1]) +(xyz[0][2]-xyz[1][2])*(xyz[0][2]-xyz[1][2]))
        print("縦:{}[m]".format(tate))
        yoko = math.sqrt((xyz[1][0]-xyz[2][0])*(xyz[1][0]-xyz[2][0]) + (xyz[1][1]-xyz[2][1])*(xyz[1][1]-xyz[2][1]) +(xyz[1][2]-xyz[2][2])*(xyz[1][2]-xyz[2][2]))
        print("横:{}[m]".format(yoko))

        np.savetxt("obj_xyz.txt", xyz)
        # # clipping_distance_in_metersm以内を画像化
        # white_color = 255 # 背景色
        # depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        # bg_removed = np.where((depth_image_3d < clipping_distance) | (depth_image_3d <= 0), white_color, color_image)
        # # 背景色となっているピクセル数をカウント
        # white_pic = np.sum(bg_removed == 255)
        # # 背景色が一定値以下になった時に、「検出」を表示する
        # if(threshold > white_pic):
        #     print("検出 {}".format(white_pic))
        # else:
        #     print("{}".format(white_pic))

        # 表示
        images = np.hstack((color_filterd_image, color_image))
        cv2.imshow('Frames', images)
        if cv2.waitKey(1) & 0xff == 27:
            break

finally:
    # ストリーミング停止
    pipeline.stop()
    cv2.destroyAllWindows()