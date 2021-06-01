"""
Usage:
------
Mouse:
    On the left image, drag with left button to show the zoom image on the right area,
    click (press) left button on the right image to select an image-point,
    and press right button to cancel the selection just before.

Keyboard:
    [c] clear plots on image
    [l] load plotted points and overlap on image
    [q] quit this application
    [s] save plotted points

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tkinter  # python3
from tkinter import filedialog as tkFileDialog  # python3


class Plotter:
    # ====================================================
    DATA_DIRPATH = './data/'  # データディレクトリのパス
    IMG_FILENAME = 'lenna.png'  # 画像ファイルの名前
    LOAD_POINTS_FILENAME = 'points.npz'  # 読み込むプロット点の npz ファイル名

    OUT_DIRPATH = './out/'  # 出力ディレクトリのパス
    OUT_POINTS_FILENAME = 'points.npz'  # 保存するプロット点の npz ファイル名

    DEFAULT_WIDTH = 350
    DEFAULT_HEIGHT = 350
    MARKER_COLOR_L = 'r'
    MARKER_COLOR_R = 'r'
    MARKER_STYLE_L = '.'
    MARKER_STYLE_R = '+'
    MARKER_SIZE_L = 3
    MARKER_SIZE_R = 15

    WILL_OPENFILE_FIRST = False
    # ---------------------------------------------------
    IMG_FILEPATH = os.path.join(DATA_DIRPATH, IMG_FILENAME)
    LOAD_POINTS_FILEPATH = os.path.join(DATA_DIRPATH, LOAD_POINTS_FILENAME)

    # ====================================================

    def __init__(self, filepath=None, imgarray=None, outDirPath=None):
        self.ax1 = None
        self.ax2 = None
        self.W = 0
        self.H = 0

        self.img = None
        self.im1 = None
        self.im2 = None

        # 初期値
        self._x1 = 0
        self._y1 = 0
        self._x2 = Plotter.DEFAULT_WIDTH
        self._y2 = Plotter.DEFAULT_HEIGHT

        self.x1 = self._x1
        self.y1 = self._y1
        self.x2 = self._x2
        self.y2 = self._y2
        self.rsize = np.min([self.x2 - self.x1, self.y2 - self.y1])

        # [debug]
        self._prev_x = 0
        self._prev_y = 0

        self.cimg = None

        self._points = []
        self.points = []

        self.PressFlag = False
        self.DragFlag = False

        self.lns = []

        self.callback = None

        if outDirPath is None:
            self.outDirPath = Plotter.OUT_DIRPATH

        if not os.path.exists(self.outDirPath):
            os.makedirs(self.outDirPath)
        self.OUT_POINTS_FILEPATH = os.path.join(self.outDirPath, Plotter.OUT_POINTS_FILENAME)

        self.Setup(filepath, imgarray)

    # -------------------------------------------------
    # マウスイベント処理
    # -------------------------------------------------
    class Callback:
        def on_plotted(self, points):
            print("current points num. = {}".format(len(points)))
        def on_quit(self):
            print("not implemented")


    def SetCallback(self, callback):
        self.callback = callback


    def Press(self, event):
        if (event.xdata is None) or (event.ydata is None):
            return

        if event.inaxes == self.ax1:
            self._x1 = event.xdata
            self._y1 = event.ydata
            self.PressFlag = True
            self.ClearScaledImg()
            plt.draw()

        if event.inaxes == self.ax2:
            x = event.xdata
            y = event.ydata
            if event.button == 1:  # 左クリック
                ix1 = int(round(self.x1))
                iy1 = int(round(self.y1))
                irsize = int(round(self.rsize))

                px = ix1 + irsize * x / Plotter.DEFAULT_WIDTH
                py = iy1 + irsize * y / Plotter.DEFAULT_HEIGHT
                if px != self._prev_x and py != self._prev_y:
                    p1, = self.ax1.plot(px, py, color=Plotter.MARKER_COLOR_L, marker=Plotter.MARKER_STYLE_L,
                                        markersize=Plotter.MARKER_SIZE_L)
                    self.points.append(p1)

                    if self.callback is not None:
                        self.callback.on_plotted(self.points);

                self._prev_x = px
                self._prev_y = py

            elif event.button == 3:  # 右クリック
                l1 = len(self.points)
                if l1 > 0:
                    p1 = self.points.pop(-1)
                    p1.remove()

                    if self.callback is not None:
                        self.callback.on_plotted(self.points);

                self.DrawScaledImg()
            plt.draw()

    def Drag(self, event):
        # global _x1, _y1, _x2, _y2, rsize, PressFlag, DragFlag
        # global cimg

        if (event.xdata is None) or (event.ydata is None):
            self.Settle()
            return

        if event.inaxes == self.ax1 and event.button == 1:
            if not self.PressFlag:
                return

            x = event.xdata
            y = event.ydata

            # 画像範囲内にあるかのチェック
            if x < 0 or x >= self.W or y < 0 or y >= self.H:
                return

            # 右下方向へのドラッグかのチェック
            if x < self._x1 or y < self._y1:
                return

            self.DragFlag = True

            self._x2 = x
            self._y2 = y
            self._x1, self._x2, _ = sorted([self._x1, self._x2, self.W])
            self._y1, self._y2, _ = sorted([self._y1, self._y2, self.H])

            _rsize = np.min([self._x2 - self._x1, self._y2 - self._y1])
            if _rsize > 0:
                _irsize = int(round(_rsize))
                self._x2 = self._x1 + _irsize
                self._y2 = self._y1 + _irsize

                ix1 = int(round(self._x1))
                iy1 = int(round(self._y1))
                ix2 = int(round(self._x2))
                iy2 = int(round(self._y2))

                self.cimg = self.img[iy1:iy2, ix1:ix2, :]

                # 画像を更新
                self.im2.set_data(self.cimg)

                # 四角形を更新
                self.DrawRect(ix1, ix2, iy1, iy2)

                # 描画
                plt.draw()

    def Settle(self):
        self.x1 = self._x1
        self.y1 = self._y1
        self.x2 = self._x2
        self.y2 = self._y2
        ix1 = int(round(self.x1))
        iy1 = int(round(self.y1))
        ix2 = int(round(self.x2))
        iy2 = int(round(self.y2))

        self.rsize = np.min([ix2 - ix1, iy2 - iy1])
        self.DrawScaledImg()
        plt.draw()

    # 離した時
    def Release(self, event):
        # global x1, y1, x2, y2, rsize
        # global PressFlag, DragFlag

        x = event.xdata
        y = event.ydata

        if event.inaxes == self.ax1 and event.button == 1:
            if self.DragFlag and x >= 0 and x < self.W and y >= 0 and y < self.H:
                self.Settle()

        elif event.inaxes == self.ax2:
            self.DrawScaledImg()
            plt.draw()

        self.PressFlag = False
        self.DragFlag = False

    # -------------------------------------------------
    # 描画処理
    # -------------------------------------------------
    # ズームされた画像（右側）から点を削除する関数
    def ClearScaledImg(self):
        while True:
            if len(self._points) == 0:
                break
            _p = self._points.pop()
            _p.remove()

    # ズームされた画像（右側）を描く関数
    def DrawScaledImg(self):
        # print("({}, {}) - ({}, {})".format(x1, y1, x2, y2))
        self.ClearScaledImg()
        for p in self.points:
            px = p._x[0]
            py = p._y[0]
            if px >= self.x1 and px < self.x2 and py >= self.y1 and py < self.y2:
                ix1 = int(round(self.x1))
                iy1 = int(round(self.y1))
                irsize = int(round(self.rsize))
                x = Plotter.DEFAULT_WIDTH * ((px - ix1) / irsize)
                y = Plotter.DEFAULT_HEIGHT * ((py - iy1) / irsize)
                _p, = self.ax2.plot(x, y, color=Plotter.MARKER_COLOR_R, marker=Plotter.MARKER_STYLE_R,
                                    markersize=Plotter.MARKER_SIZE_R)
                self._points.append(_p)

    # 四角形を描く関数
    def ClearPoints(self):
        while True:
            if len(self._points) == 0:
                break
            _p = self._points.pop()
            _p.remove()
        while True:
            if len(self.points) == 0:
                break
            p = self.points.pop()
            p.remove()

    def LoadPoints(self, filepath):
        # global points
        data = np.load(filepath)
        imgPoints = data['points']
        print(imgPoints)
        self.points = []
        for p in imgPoints:
            px = p[0]
            py = p[1]
            p1, = self.ax1.plot(px, py, color=Plotter.MARKER_COLOR_L, marker=Plotter.MARKER_STYLE_L,
                                markersize=Plotter.MARKER_SIZE_L)
            self.points.append(p1)

    # 四角形を描く関数
    def DrawRect(self, x1, x2, y1, y2):
        rect = [[[x1, x2], [y1, y1]],
                [[x2, x2], [y1, y2]],
                [[x1, x2], [y2, y2]],
                [[x1, x1], [y1, y2]]]
        for i, r in enumerate(rect):
            self.lns[i].set_data(r[0], r[1])

    # -------------------------------------------------
    # キーイベント処理
    # -------------------------------------------------
    def onKey(self, event):
        if event.key == 'c':
            print("clear points")
            self.ClearPoints()
            plt.draw()
        elif event.key == 'l':
            print("loaded points below")
            print("-------------------")
            self.ClearPoints()
            self.LoadPoints(Plotter.LOAD_POINTS_FILEPATH)
            print("-------------------")
            self.DrawScaledImg()
            plt.draw()
        elif event.key == 's':
            imgPoints = self.GetImagePointsArray()
            print("saved points below")
            print("------------------")
            print(imgPoints)
            print("------------------")
            np.savez(Plotter.OUT_POINTS_FILEPATH, points=imgPoints)
        elif event.key == 'q':
            if self.callback is not None:
                self.callback.on_quit();


    def GetImagePointsArray(self):
        imgPointsList = []
        for p in self.points:
            px = p._x[0]
            py = p._y[0]
            imgPointsList.extend([px, py])
        imgPoints = np.array(imgPointsList)
        imgPoints = np.reshape(imgPoints, (-1, 2))
        return imgPoints


    def Setup(self, filepath=None, imgarray=None):
        # ここから本処理
        is_file_opened = False

        if imgarray is None:
            if filepath is None:
                if Plotter.WILL_OPENFILE_FIRST:
                    root = tkinter.Tk()  # python3
                    root.withdraw()

                    fTyp = [('Image File', '*.png'), ('Image File', '*.jpg'), ('Image File', '*.jpeg')]
                    iDir = os.path.dirname(__file__)
                    filename = tkFileDialog.askopenfilename(filetypes=fTyp, initialdir=iDir)
                    if not (filename == ''):
                        filepath = os.path.join(os.path.dirname(__file__), filename)
                        is_file_opened = True

                if not is_file_opened:
                    filepath = os.path.join(os.path.dirname(__file__), Plotter.IMG_FILEPATH)

            if not os.path.exists(filepath):
                print("file: {} is not found.".format(filepath))
                return

            _img = Image.open(filepath)
        else:
            _img = imgarray

        # numpy.ndarrayに
        self.img = np.asarray(_img)

        self.H = self.img.shape[0]
        self.W = self.img.shape[1]
        print("image size: (W, H) = ({}, {})".format(self.W, self.H))

        ix1, ix2 = sorted([self.x1, self.x2])
        iy1, iy2 = sorted([self.y1, self.y2])

        # 画像の一部を抜き出す
        self.cimg = self.img[iy1:iy2, ix1:ix2, :]

        # plot
        # plt.close('all')
        plt.figure(figsize=(8, 4))

        # subplot 1
        self.ax1 = plt.subplot(1, 2, 1)

        self.im1 = plt.imshow(self.img, cmap='gray')

        # 四角形を描画
        rect = [[[self.x1, self.x2], [self.y1, self.y1]],
                [[self.x2, self.x2], [self.y1, self.y2]],
                [[self.x1, self.x2], [self.y2, self.y2]],
                [[self.x1, self.x1], [self.y1, self.y2]]]

        self.lns = []
        for r in rect:
            ln, = plt.plot(r[0], r[1], color='r', lw=2)
            self.lns.append(ln)

        plt.axis('off')

        # 拡大図
        self.ax2 = plt.subplot(1, 2, 2)
        self.im2 = plt.imshow(self.cimg, cmap='gray')

        plt.clim(self.im1.get_clim())
        plt.axis('off')

        # イベント
        plt.rcParams['keymap.zoom'] = ''  # 'o'
        plt.rcParams['keymap.save'] = ''  # 's'
        plt.rcParams['keymap.xscale'] = ''  # 'L'
        plt.rcParams['keymap.yscale'] = ''  # 'l'

        plt.connect('button_press_event', self.Press)
        plt.connect('motion_notify_event', self.Drag)
        plt.connect('button_release_event', self.Release)
        plt.connect('key_press_event', self.onKey)

    def show(self, adjust=True):
        if adjust:
            fm = plt.get_current_fig_manager()
            geom = fm.window.geometry()
            x, y, dx, dy = geom.getRect()

            margin_xy = 50
            fm.window.setGeometry(0 + margin_xy, y + margin_xy, dx, dy)
        plt.show()

# メインの処理
if __name__ == '__main__':
    app = Plotter()
    app.show()
