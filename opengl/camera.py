import math
import numpy as np
import cv2

import OpenGL.GL as gl
import OpenGL.GLU as glu
import OpenGL.GLUT as glut
import glfw
import ctypes

from mouse import MouseLister

class GLCamera:
    def __init__(self,
                 display_size, display_title="window",
                 zNear=0.001, zFar=100.0, fovy=20.0):
        self.window = None
        self.display_size = display_size
        self.zNear = zNear
        self.zFar = zFar
        self.fovy = fovy

        # カメラの位置姿勢
        self.tvec = np.array([0, 0, 0], np.float32)
        self.rvec = np.array([0, 0, 0], np.float32)

        self.roll = math.radians(0)
        self.pitch = math.radians(0)
        self.yaw = math.radians(0)

        self.modelview_matrix = None
        self.projection_matrix = None

        self.objects = {}
        self.textures = {}
        self.stop_flag = False

        self.on_draw = []
        self.on_keyboard_event = []
        self.on_mouse_button = []
        self.on_mouse_drag = []
        self.on_mouse_scroll = []

        self.create_window(display_title=display_title)

    def create_window(self, display_title="camera view"):
        # Create a windowed mode window and its OpenGL context
        (w, h) = self.display_size
        self.window = glfw.create_window(w, h, display_title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError('Could not create an window')

        # Make the window's context current
        glfw.make_context_current(self.window)

        # window callback
        glfw.set_window_size_callback(self.window, self._windows_resize)
        glfw.set_window_refresh_callback(self.window, self._window_refresh)

        # keyboard callback
        glfw.set_key_callback(self.window, self._on_keyboard_event)

        # MouseListener
        mouse_listener = MouseLister(self.window)
        mouse_listener.on_mouse_button = self._on_mouse_button
        mouse_listener.on_mouse_drag = self._on_mouse_drag
        mouse_listener.on_mouse_scroll = self._on_mouse_scroll

        GLCamera.setup_opengl(self.window)

    def is_available(self):
        return not glfw.window_should_close(self.window)

    def _draw(self):
        for l in self.on_draw:
            l()

    def refresh(self):
        self._window_refresh(self.window)

    def projection(self):
        _w, _h = self.display_size
        gl.glViewport(0, 0, _w, _h)

        # 射影行列の設定
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()

        aspect = _w / float(_h)
        top = self.zNear * np.tan(np.radians(self.fovy))
        bottom = -self.zNear * np.tan(np.radians(self.fovy))
        left = - top * aspect
        right = top * aspect

        pm = (gl.GLfloat * 16)()
        pm[0] = 2 * self.zNear / (right - left)
        pm[5] = 2 * self.zNear / (top - bottom)
        pm[8] = (right + left) / (right - left)
        pm[9] = (top + bottom) / (top - bottom)
        pm[10] = - (self.zFar + self.zNear) / (self.zFar - self.zNear)
        pm[11] = - 1
        pm[14] = - 2 * self.zFar * self.zNear / (self.zFar - self.zNear)

        gl.glLoadMatrixf((ctypes.c_float * 16)(*pm))
        self.projection_matrix = np.array(pm).reshape(4, 4).transpose()

    def modelview(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        # カメラの回転・並進移動
        _matrix = GLCamera.rotation_matrix_rpy_euler(self.roll, self.pitch, self.yaw)
        _matrix[12] = self.tvec[0]
        _matrix[13] = self.tvec[1]
        _matrix[14] = self.tvec[2]
        _matrix[15] = 1
        gl.glLoadMatrixf((ctypes.c_float * 16)(*_matrix))

        # 視点の設定（デフォルトの座標系なのでなくても同じ）
        # glu( float eyeX, float eyeY, float eyeZ,
        #            float centerX, float centerY, float centerZ,
        #            float upX, float upY, float upZ)
        glu.gluLookAt(0.0, 1.0, 0.0,
                      0.0, 0.0, -8.0,
                      0.0, 1.0, 0.0)

        # -------------------------------------
        # numpy 配列への変換（最後に転置する）
        mm = (gl.GLfloat * 16)()
        gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX, mm)
        self.modelview_matrix = np.array(mm).reshape(4, 4).transpose()

        # -------------------------------------
        # 回転ベクトルへの変換
        R = self.modelview_matrix[0:3, 0:3]
        self.rvec, _ = cv2.Rodrigues(R)
        self.rvec = self.rvec.reshape(3)

    def _window_refresh(self, window):
        glfw.make_context_current(self.window)

        gl.glClearColor(0.0, 0.0, 0.0, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        gl.glLoadIdentity()

        self.projection()
        self.modelview()

        for o in self.objects.values():
            o.draw()

        for t in self.textures.values():
            t.draw()

        # Swap front and back buffers
        glfw.swap_buffers(window)

    def _windows_resize(self, window, w, h):
        (_w, _h) = self.display_size
        print("windows resized = ({}, {}) --> ({}, {})".format(_w, _h, w, h))
        self.display_size[0] = w
        self.display_size[1] = h

    # ===============================================================
    # Mouse Events
    # ---------------------------------------------------------------
    def _on_mouse_button(self, l, button, action, mods, x, y):
        # print("button: ({}, {})".format(x, y))
        for l in self.on_mouse_button:
            l(l, button, action, mods, x, y)
        self._draw()

    def _on_mouse_drag(self, l, x, y, dx, dy):
        if l.pressed:
            # print("drag: ({}, {}) -> ({}, {}) | (dx, dy) = ({}, {})".format(x-dx, y-dy, x, y, dx, dy))
            if l.mouse_btns[0]:
                self.yaw -= dx * 0.005
                self.pitch -= dy * 0.005
            if l.mouse_btns[1]:
                self.tvec += np.array((dx, dy, 0)) * 0.005
            if l.mouse_btns[2]:
                self.roll -= dx * 0.005
        for l in self.on_mouse_drag:
            l(l, x, y, dx, dy)
        self._draw()

    def _on_mouse_scroll(self, l, xoffset, yoffset):
        # print('scroll: ')
        dz = yoffset * 0.1
        self.tvec[2] += dz
        for l in self.on_mouse_scroll:
            l(l, xoffset, yoffset)
        self._draw()


    # ===============================================================
    # Keyboard Events
    # ---------------------------------------------------------------
    def _on_keyboard_event(self, window, key, scancode, action, mods):
        for l in self.on_keyboard_event:
            l(window, key, scancode, action, mods)

    # ===============================================================
    # class method
    # ---------------------------------------------------------------
    @classmethod
    def setup_opengl(cls, window):
        glfw.make_context_current(window)

        gl.glEnable(gl.GL_POINT_SMOOTH)
        gl.glHint(gl.GL_POINT_SMOOTH_HINT, gl.GL_NICEST)
        gl.glEnable(gl.GL_LINE_SMOOTH)
        gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)

        diffuse = (0.5, 0.5, 0.5, 1.0)
        ambient = (0.1, 0.1, 0.1, 1.0)
        specular = (0.1, 0.1, 0.1, 1.0)

        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (1.5, 1.5, 1.0, 1.0))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, diffuse)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, ambient)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_SPECULAR, specular)
        gl.glEnable(gl.GL_LIGHT0)

        gl.glLightfv(gl.GL_LIGHT1, gl.GL_POSITION, (-1.5, 1.5, 1.0, 1.0))
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_DIFFUSE, diffuse)
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_AMBIENT, ambient)
        gl.glLightfv(gl.GL_LIGHT1, gl.GL_SPECULAR, specular)
        gl.glEnable(gl.GL_LIGHT1)

        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_DEPTH_TEST)


    # [see] 外因性オイラー角でのロール・ピッチ・ヨーから回転行列への変換
    @classmethod
    def rotation_matrix_rpy_euler(cls, roll, pitch, yaw):
        sr = np.sin(roll)
        sp = np.sin(pitch)
        sy = np.sin(yaw)
        cr = np.cos(roll)
        cp = np.cos(pitch)
        cy = np.cos(yaw)

        rm = (gl.GLfloat * 16)()
        rm[0] = sp * sr * sy + cr * cy
        rm[1] = sr * cp
        rm[2] = sp * sr * cy - sy * cr
        rm[3] = 0
        rm[4] = sp * sy * cr - sr * cy
        rm[5] = cp * cr
        rm[6] = sp * cr * cy + sr * sy
        rm[7] = 0
        rm[8] = sy * cp
        rm[9] = -sp
        rm[10] = cp * cy
        rm[11] = 0
        rm[12] = 0
        rm[13] = 0
        rm[14] = 0
        rm[15] = 1
        return rm


    # 回転行列から外因性オイラー角でのロール・ピッチ・ヨーへの変換
    @classmethod
    def rotation_matrix_2_rpy_euler_angle(cls, R, delta=1e-5):
        roll, pitch, yaw = 0, 0, 0
        if np.abs(R[1, 2] + 1.0) < delta:  # sin(alpha) == -1 の場合 beta と gamma は一意に定まらないので，beta = 0 とする．
            roll = np.arctan2(R[2, 0], R[2, 1])
            pitch = - np.pi / 2.0
            yaw = 0.0
        elif np.abs(R[1, 2] - 1.0) < delta:  # sin(alpha) == 1 の場合 beta と gamma は一意に定まらないので，beta = 0 とする．
            roll = np.arctan2(R[2, 0], R[2, 1])
            pitch = np.pi / 2.0
            yaw = 0.0
        else:
            roll = np.arctan2(R[1, 0], R[1, 1])
            pitch = -np.arcsin(R[1, 2])
            yaw = np.arctan2(R[0, 2], R[2, 2])

        return roll, pitch, yaw

