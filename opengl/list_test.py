import numpy as np

bb = np.loadtxt("bb_cord.txt")
BOARD_X = 0.
BOARD_Y = 0.
BOARD_WIDTH = 0.8
BOARD_HEIGHT = 0.45
board_vertices = ((BOARD_X - BOARD_WIDTH / 2, BOARD_Y + BOARD_HEIGHT),)
print(board_vertices[0])
print(bb)
print(bb[0][1], bb[1][0])
print(type(bb))