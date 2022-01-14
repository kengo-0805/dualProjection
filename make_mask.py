import cv2
import numpy as np
# 内側塗り潰し
img = cv2.imread("pj1.png")
pj1_cord = np.loadtxt("pj1sh_cord.txt")
pts = [np.array(((int(pj1_cord[0][0]), int(pj1_cord[0][1])), (int(pj1_cord[1][0]), int(pj1_cord[1][1])), (int(pj1_cord[2][0]), int(pj1_cord[2][1])), (int(pj1_cord[3][0]), int(pj1_cord[3][1]))))]
color = [0, 0, 0]
cv2.fillPoly(img, pts, color)
# cv2.polylines(img, pts, True, color, thickness=2)
cv2.imwrite("pj1_after.png", img)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# 外側塗り潰し
img = cv2.imread("pj2.png") 
stencil = np.zeros(img.shape).astype(img.dtype) 
pj2_cord = np.loadtxt("pj2sh_cord.txt")
contours = [np.array(((int(pj2_cord[0][0]), int(pj2_cord[0][1])), (int(pj2_cord[1][0]), int(pj2_cord[1][1])), (int(pj2_cord[2][0]), int(pj2_cord[2][1])), (int(pj2_cord[3][0]), int(pj2_cord[3][1]))))]
color = [255, 255, 255] 
cv2.fillPoly(stencil, contours, color) 
# cv2.polylines(img, contours, True, color, thickness=2)
result = cv2.bitwise_and(img, stencil) 
cv2.imwrite("pj2_after.png", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows() 