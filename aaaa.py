import cv2
import numpy as np

img = cv2.imread("pj1.png")
bb = np.loadtxt("pjjjj_cord.txt")
cv2.rectangle(img, (int(bb[0][0]), int(bb[0][1])), (int(bb[2][0]), int(bb[2][1])), color=(0, 255, 0), thickness=2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()