import glfw

class MouseLister:

    def  __init__(self, window):
        self.window = window
        self.pressed = False
        self.released = False

        # left, right, middle
        self.mouse_btns = [False, False, False]

        self.last_x = -1
        self.last_y = -1

        self.on_mouse_button = None
        self.on_mouse_drag = None
        self.on_mouse_scroll = None

        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_mouse_drag)
        glfw.set_scroll_callback(self.window, self._on_mouse_scroll)


    def _on_mouse_button(self, window, button, action, mods):
        """
        mods:
            glfw.MOD_SHIFT
            glfw.MOD_CONTROL
            glfw.MOD_ALT
            glfw.MOD_SUPER
            glfw.MOD_CAPS_LOCK
            glfw.MOD_NUM_LOCK
       """
        (x, y) = glfw.get_cursor_pos(window)
        self.last_x = x
        self.last_y = y
        self.pressed = (action == glfw.PRESS)
        self.released = (action == glfw.RELEASE)
        if self.pressed or self.released:
            self.mouse_btns[0] = self.pressed & (button == glfw.MOUSE_BUTTON_LEFT)
            self.mouse_btns[1] = self.pressed & (button == glfw.MOUSE_BUTTON_RIGHT)
            self.mouse_btns[2] = self.pressed & (button == glfw.MOUSE_BUTTON_MIDDLE)
        if self.on_mouse_button is not None:
            self.on_mouse_button(self, button, action, mods, x, y)

    def _on_mouse_drag(self, window, x, y):
        dx = x - self.last_x
        dy= y - self.last_y
        self.last_x = x
        self.last_y = y
        if self.on_mouse_drag is not None:
            self.on_mouse_drag(self, x, y, dx, dy)

    def _on_mouse_scroll(self, window, xoffset, yoffset):
        if self.on_mouse_scroll is not None:
            self.on_mouse_scroll(self, xoffset, yoffset)