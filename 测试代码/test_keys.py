"""4 按键 + OLED 回显测试"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Drivers.SSD1306 import SSD1306_IIC

oled = SSD1306_IIC(7, with_keys=True)
oled.show_text('press key')
try:
    while True:
        k = oled.wait_key(timeout_ms=1000)
        if k is None:
            continue
        print(k)
        oled.show_text(k, x=55, y=15, size=32)
except KeyboardInterrupt:
    pass
finally:
    oled.close()
