# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2
# import open3d as o3d

WIDTH = 640
HEIGHT = 480
THRESHOLD = 0.2 # これより近い距離の画素を無視する
SCREEN = 0.4
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

# 距離[m] = depth * depth_scale
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
        

        # D画像のフレームから画素値をnumpy配列に変換
        depth_image = np.asanyarray(depth_frame.get_data()) # 深度の画素値が入っている


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
                if THRESHOLD < dist and dist < SCREEN + 0.05: # 閾値以上スクリーン以下であれば
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
        # 形式変換
        # hsv = cv2.cvtColor(color_filterd_image, cv2.COLOR_BGR2HSV)
        # cv2.imshow("hsv image",hsv)
        binary = cv2.inRange(color_filterd_image, (0, 254, 0), (0, 255, 0))
        cv2.imshow("niti",binary)
        # 輪郭抽出
        contours = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
        # 面積が一定以上の輪郭のみ残す。
        area_thresh = 10000
        contours = list(filter(lambda x: cv2.contourArea(x) > area_thresh, contours)) #xを与えてそのエリアが閾値より大きければリスト
        # # 輪郭を矩形で囲む。
        for cnt in contours:
            # 輪郭に外接する長方形を取得する。
            x, y, width, height = cv2.boundingRect(cnt)
            # 描画する。
            cv2.rectangle(color_image, (x, y), (x + width, y + height), color=(0, 255, 0), thickness=2)
            # cv2.polylines(color_image, cnt, True, (255, 0, 0), 5)
            print("左上:{},{}".format(x,y))
            print("右下:{},{}".format(x+width,y+height))


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