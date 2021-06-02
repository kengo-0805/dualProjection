}^# -*- coding: utf-8 -*-
import pyrealsense2 as rs
import numpy as np
import cv2

WIDTH = 640
HEIGHT = 480

# ストリーミング初期化
config = rs.config()
config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)

# ストリーミング開始
pipeline = rs.pipeline()
profile = pipeline.start(config)

# 距離[m] = depth * depth_scale 
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
clipping_distance_in_meters = 0.4 # 40cm以内を検出
clipping_distance = clipping_distance_in_meters / depth_scale

# Alignオブジェクト生成
align_to = rs.stream.color
align = rs.align(align_to)

threshold = (WIDTH * HEIGHT * 3) * 0.95

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not depth_frame or not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # clipping_distance_in_metersm以内を画像化
        white_color = 255 # 背景色
        depth_image_3d = np.dstack((depth_image, depth_image, depth_image))
        bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), white_color, color_image)
        # 背景色となっているピクセル数をカウント
        white_pic = np.sum(bg_removed == 255)
        # 背景色が一定値以下になった時に、「検出」を表示する
        if(threshold > white_pic):
            print("検出 {}".format(white_pic))
        else:
            print("{}".format(white_pic))

        images = np.hstack((bg_removed, color_image))
        cv2.imshow('Frames', images)

        if cv2.waitKey(1) & 0xff == 27:
            break

finally:
    # ストリーミング停止
    pipeline.stop()
    cv2.destroyAllWindows()