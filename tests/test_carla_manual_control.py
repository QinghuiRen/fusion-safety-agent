from types import SimpleNamespace

from fusion_tracking_carla.session import ManualKeyboardController


class _Events:
    def __init__(self):
        self.items = []

    def get(self):
        items, self.items = self.items, []
        return items


class _FakePygame:
    QUIT = 1
    KEYDOWN = 2
    KEYUP = 3
    WINDOWFOCUSLOST = 4
    K_ESCAPE = 27
    K_q = 113
    K_w = 119
    K_s = 115
    K_a = 97
    K_d = 100
    K_UP = 273
    K_DOWN = 274
    K_LEFT = 276
    K_RIGHT = 275
    K_SPACE = 32

    def __init__(self):
        self.event = _Events()


def test_keydown_is_latched_until_keyup():
    pygame = _FakePygame()
    controller = ManualKeyboardController(pygame, use_global_keyboard=False)
    pygame.event.items = [SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_w)]
    first, _ = controller.read(0.05)
    held, _ = controller.read(0.05)
    pygame.event.items = [SimpleNamespace(type=pygame.KEYUP, key=pygame.K_w)]
    released, _ = controller.read(0.05)
    assert first["throttle"] == 1.0
    assert held["throttle"] == 1.0
    assert released["throttle"] == 0.0


def test_focus_loss_clears_latched_control():
    pygame = _FakePygame()
    controller = ManualKeyboardController(pygame, use_global_keyboard=False)
    pygame.event.items = [SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_d)]
    steering, _ = controller.read(0.05)
    assert steering["steer"] > 0.0
    pygame.event.items = [SimpleNamespace(type=pygame.WINDOWFOCUSLOST)]
    cleared, _ = controller.read(0.05)
    assert cleared["throttle"] == 0.0
    assert abs(cleared["steer"]) < abs(steering["steer"])


def test_q_toggles_reverse_once_per_keydown():
    pygame = _FakePygame()
    controller = ManualKeyboardController(pygame, use_global_keyboard=False)
    pygame.event.items = [SimpleNamespace(type=pygame.KEYDOWN, key=pygame.K_q)]
    reverse, _ = controller.read(0.05)
    held, _ = controller.read(0.05)
    assert reverse["reverse"] is True
    assert held["reverse"] is True


class _FakeAsyncKeyboard:
    def __init__(self):
        self.pressed = set()

    def down(self, key):
        return key in self.pressed


def test_windows_async_backend_reads_w_without_pygame_focus():
    pygame = _FakePygame()
    keyboard = _FakeAsyncKeyboard()
    controller = ManualKeyboardController(pygame, async_key_reader=keyboard)
    keyboard.pressed.add(0x57)  # Windows VK_W
    control, _ = controller.read(0.05)
    assert control["throttle"] == 1.0
    assert controller.backend_name == "Windows GetAsyncKeyState (global)"
