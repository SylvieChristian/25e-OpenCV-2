"""OLED 菜单测试 — 绑定到 DetectorConfig 部分字段 + 模拟实时 FPS"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Drivers.SSD1306 import SSD1306_IIC
from Drivers.OLED_Menu import Menu
from Algorithm.Config import DetectorConfig


def main():
    cfg = DetectorConfig()
    fps = [0.0]
    t0 = time.time()
    frame = [0]

    oled = SSD1306_IIC(7, with_keys=True)
    menu = Menu(oled, [
        {'name': 'FPS ',  'get': lambda: fps[0]},                                # 只读实时
        {'name': 'Task', 'get': lambda: cfg.task_mode,
                          'set': lambda v: setattr(cfg, 'task_mode', v),
                          'choices': ['normal', 'ellipse']},
        {'name': 'Area', 'get': lambda: cfg.area_min,
                          'set': lambda v: setattr(cfg, 'area_min', v),
                          'range': (0, 8000, 100)},
        {'name': 'P2',   'get': lambda: cfg.hough_param2,
                          'set': lambda v: setattr(cfg, 'hough_param2', round(v, 2)),
                          'range': (0.0, 1.0, 0.05)},
        {'name': 'CLAHE','get': lambda: cfg.use_clahe,
                          'set': lambda v: setattr(cfg, 'use_clahe', v),
                          'choices': [False, True]},
        {'name': '<DUMP>', 'action': lambda: print('cfg:', cfg)},
    ])

    try:
        while True:
            menu.step(timeout_ms=80)
            frame[0] += 1
            now = time.time()
            if now - t0 >= 0.5:
                fps[0] = frame[0] / (now - t0)
                frame[0] = 0; t0 = now
    except KeyboardInterrupt:
        pass
    finally:
        oled.close()


if __name__ == '__main__':
    main()
