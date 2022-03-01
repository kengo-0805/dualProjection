import cv2
import os
import tkinter as tk
from realsensecv import RealsenseCapture

global n

cap_ob = RealsenseCapture()
# プロパティの設定
cap_ob.WIDTH = 1280
cap_ob.HEIGHT = 720
cap_ob.FPS = 30
cap_ob.start()

def save_frame_camera_key(ext='png', delay=1):
    global n
    # cap_ob = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # cap_ex = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    # if not cap_ex.isOpened():
    # return

    # os.makedirs('data/temp/observer', exist_ok=True)
    base_path_ob = os.path.join('cap', 'cap')
    # base_path_ex = os.path.join('data/temp/external', 'external')

    n = 0
    while True:
        ret_ob, frame_ob = cap_ob.read()
        frame_ob = frame_ob[0]
        # ret_ex, frame_ex = cap_ex.read()
        cv2.imshow("frame_ob", frame_ob)
        # cv2.imshow("frame_ex", frame_ex)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('c'):
            cv2.imwrite('{}_{:02}.{}'.format(base_path_ob, n, ext), frame_ob)
            # cv2.imwrite('{}_{:02}.{}'.format(base_path_ex, n, ext), frame_ex)

            n += 1
        elif key == ord('q'):
            break

    cv2.destroyAllWindows()

# image_projection()
save_frame_camera_key()