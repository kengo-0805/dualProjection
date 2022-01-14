import cv2
import numpy as np

img = cv2.imread("./q.png")
height, width, channels = img.shape[:3]

pts = np.array(((551, 80), (614, 740), (608, 743)))
for i in range(4):
  for i in range(2):
    if pts[i][i] > width:
      pts[i][i] = width
    if pts[i][i] < 0:
      pts[i][i] = 0
  if pts[i][i] > height or pts[i][i] <0:
    pts[i][i] = height
  if pts[i][i] < 0:
    pts[i][i] = 0
print("h:{},w:{}".format(height, width))
print(pts)
cv2.fillPoly(img,
              [pts],
              [0, 255, 0])
cv2.imwrite("./chess_after.png",img)