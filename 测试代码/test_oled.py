"""
OLED 1306 点亮测试 — 只画一帧，画完留在屏上
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Drivers.SSD1306 import SSD1306_IIC
from PIL import Image, ImageDraw, ImageFont

I2C_BUS = 7
FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'

def main():
    print(f'[*] init SSD1306 on i2c-{I2C_BUS} ...')
    oled = SSD1306_IIC(I2C_BUS)

    img = Image.new('1', (128, 64), 0)
    draw = ImageDraw.Draw(img)
    font_big = ImageFont.truetype(FONT_PATH, 20) if os.path.exists(FONT_PATH) else ImageFont.load_default()
    font_small = ImageFont.truetype(FONT_PATH, 14) if os.path.exists(FONT_PATH) else ImageFont.load_default()
    draw.text((0, 0), 'Mamba out!', fill=1, font=font_big)
    draw.text((0, 30), '你好 Sylvie', fill=1, font=font_small)
    draw.rectangle([(0, 60), (127, 63)], outline=1, fill=1)

    print('[*] render one frame ...')
    oled.renderPillowImage(img)
    print('[*] done. (NO clear, NO close — frame stays on screen)')

if __name__ == '__main__':
    main()
