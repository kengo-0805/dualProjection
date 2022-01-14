import cv2 
import numpy as np

img1 = cv2.imread("pj1.png")
img2 = cv2.imread("chessboard.png")
img3 = cv2.imread("pj2.png")

from matplotlib import pyplot as plt

# #パラメータ指定
# max_pts=500
# good_match_rate=0.5
# min_match=10

# orb = cv2.ORB_create(max_pts)

# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)


# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# matches = bf.match(des1, des2)

# matches = sorted(matches, key=lambda x: x.distance)
# good = matches[:int(len(matches) * good_match_rate)]

# match_img = cv2.drawMatches(img1,kp1,img2,kp2,good, None,flags=2)
# cv2.imshow("match_img",match_img)
# cv2.waitKey(0)
# cv2.imwrite('match_img.png',match_img)
pj1_cord = np.loadtxt("pj1_cord.txt")
pj2_cord = np.loadtxt("pj2_cord.txt")
dst_pts = np.asarray(((int(pj1_cord[0][0]), int(pj1_cord[0][1])), (int(pj1_cord[1][0]), int(pj1_cord[1][1])), (int(pj1_cord[2][0]), int(pj1_cord[2][1])), (int(pj1_cord[3][0]), int(pj1_cord[3][1]))))
src_pts = np.asarray([(0, 0), (0, 720), (1280, 720), (1280, 0)])
h, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
dst_pts2 = np.asarray([[int(pj2_cord[0][0]), int(pj2_cord[0][1])], [int(pj2_cord[1][0]), int(pj2_cord[1][1])], [int(pj2_cord[2][0]), int(pj2_cord[2][1])], [int(pj2_cord[3][0]), int(pj2_cord[3][1])]])
h2, mask2 = cv2.findHomography(dst_pts2, src_pts, cv2.RANSAC)

# if len(good) > min_match:
    # src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    # dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    # h, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC)
print("homo1\n", h)
print("homo2\n", h2)

height, width ,channels= img1.shape[:3]
height, width = (720, 1280)
dst_img = cv2.warpPerspective(img3, h2, (width, height))

cv2.imshow("dst_img",dst_img)
cv2.waitKey(0)

cv2.imwrite('dst_img.png',dst_img)
# else:
# cv2.imshow("dst_img",img1)
# cv2.waitKey(0)