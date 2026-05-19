"""
SSD1306 OLED (128x64, I2C) 四按键驱动
show_text / show_lines / set_font 
"""
import os

import smbus2 as smbus
import numpy as np
from PIL import Image, ImageDraw, ImageFont


#默认字体候选列表
DEFAULT_FONT_CANDIDATES = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/arphic/uming.ttc',
    'wqy-microhei.ttc',
]


class SSD1306_IIC():
    def __init__(self, iicDeviceIndex, address=0x3c, font_path=None, with_keys=False):
        self.IICInterface = smbus.SMBus(iicDeviceIndex)
        self.displayWidth = 128
        self.displayHeight = 64
        self.address = address
        self.framebuffer = np.zeros([1024,], dtype=np.uint8)
        self._font_path = font_path or self._auto_font()
        self._font_cache = {}
        self.initProcess()

        self.keys = None
        if with_keys:
            from Drivers.Key import Keys
            self.keys = Keys()

    def clear(self,white=False):
        for i in range(1024):
            self.framebuffer[i] = 0xff if white else 0x00
        self.updateScreen()

    def close(self):
        if self.keys:
            self.keys.close()
        self.IICInterface.close()

    def writeCommand(self,cmd):
        self.IICInterface.write_byte_data(self.address, 0x00, cmd)

    def writeData(self,data):
        self.IICInterface.write_byte_data(self.address, 0x40, data)

    def renderSinglePixel(self,x,y,value):
        if x < 0 or x >= self.displayWidth: return
        elif y < 0 or y >= self.displayHeight: return
        part = self.framebuffer[x + (y // 8) * self.displayWidth]
        if value:
            part |= 1 << (y%8)
        else:
            part &= ~(1<<(y%8))
        self.framebuffer[x + (y // 8) * self.displayWidth] = part

    def renderPillowImage(self, pillowImage):
        assert type(pillowImage) == Image.Image
        assert pillowImage.mode == '1'
        array = np.array(pillowImage)
        # print(array)
        for x in range(self.displayWidth):
            for y in range(self.displayHeight):
                self.renderSinglePixel(x,y,1 if array[y][x] else 0)
                # self.renderSinglePixel(x,y, array[y][x])
        self.updateScreen()

    def updateScreen(self):
        for i in range(8):
            self.writeCommand(0xb0+i)
            self.writeCommand(0x00)
            self.writeCommand(0x10)
            for j in range(self.displayWidth):
                self.writeData(self.framebuffer[i*self.displayWidth + j])

    def initProcess(self):
        self.writeCommand(0xae)
        self.writeCommand(0x20)
        self.writeCommand(0x10)
        self.writeCommand(0xb0)
        self.writeCommand(0xc8)
        self.writeCommand(0x00)
        self.writeCommand(0x10)
        self.writeCommand(0x40)
        self.writeCommand(0x81)
        self.writeCommand(0xff)
        self.writeCommand(0xa1)
        self.writeCommand(0xa6)
        self.writeCommand(0xa8)
        self.writeCommand(self.displayHeight - 1)
        self.writeCommand(0xa4)
        self.writeCommand(0xd3)
        self.writeCommand(0x00)
        self.writeCommand(0xd5)
        self.writeCommand(0xf0)
        self.writeCommand(0xd9)
        self.writeCommand(0x22)

        self.writeCommand(0xda)
        self.writeCommand(0x12)

        self.writeCommand(0xdb)
        self.writeCommand(0x40)
        self.writeCommand(0x8d)
        self.writeCommand(0x14)
        self.writeCommand(0xaf)

    # ====================== 新增便捷方法 ======================

    @staticmethod
    def _auto_font():
        for p in DEFAULT_FONT_CANDIDATES:
            if os.path.exists(p):
                return p
        return None

    def set_font(self, path):
        self._font_path = path
        self._font_cache.clear()

    def _font(self, size):
        if size not in self._font_cache:
            if self._font_path:
                self._font_cache[size] = ImageFont.truetype(self._font_path, size)
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _blank_image(self):
        return Image.new('1', (self.displayWidth, self.displayHeight), 0)

    def show_text(self, text, x=0, y=0, size=14):
        img = self._blank_image()
        ImageDraw.Draw(img).text((x, y), str(text), fill=1, font=self._font(size))
        self.renderPillowImage(img)

    def show_lines(self, *lines, sizes=None, padding=2, x=0, y=0):
        img = self._blank_image()
        draw = ImageDraw.Draw(img)
        if sizes is None:
            sizes = [14] * len(lines)
        elif isinstance(sizes, int):
            sizes = [sizes] * len(lines)
        cur_y = y
        for line, sz in zip(lines, sizes):
            draw.text((x, cur_y), str(line), fill=1, font=self._font(sz))
            cur_y += sz + padding
        self.renderPillowImage(img)

    def wait_key(self, timeout_ms=None):
        return self.keys.get(timeout_ms) if self.keys else None
