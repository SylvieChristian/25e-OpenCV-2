"""OLED 通用菜单 — 按键导航 + 实时显示 + 参数编辑
item = dict:
    name:    str            显示名
    get:     callable()     读当前值；只读项只给 get
    set:     callable(v)    写新值；与 range/choices 配对
    range:   (min,max,step) 数值编辑
    choices: list           选项编辑
    action:  callable()     触发型（不带 get/set）
按键：^v 浏览或调值，# 进入/退出编辑或触发动作，* 取消编辑
"""

class Menu:
    def __init__(self, oled, items, visible=4, size=13):
        self.oled = oled
        self.items = items
        self.visible = visible
        self.size = size
        self.cur = 0            #索引
        self.top = 0            #第一行
        self.edit = False       #编辑状态

    def __getitem__(self, name):
        for it in self.items:
            if it['name'] == name:
                return it['get']()
        raise KeyError(name)

    def step(self, timeout_ms=100):
        k = self.oled.wait_key(timeout_ms=timeout_ms)
        if k:
            self._handle(k)
        self._draw()

    def _handle(self, k):
        it = self.items[self.cur]
        if self.edit:
            if k in ('#', '*'):
                self.edit = False
            elif 'range' in it:
                lo, hi, step = it['range']
                v = it['get']()
                if k == '^': v = min(hi, v + step)
                if k == 'v': v = max(lo, v - step)
                it['set'](v)
            elif 'choices' in it:
                ch = it['choices']; v = it['get']()
                i = ch.index(v) if v in ch else 0
                if k == '^': i = (i + 1) % len(ch)
                if k == 'v': i = (i - 1) % len(ch)
                it['set'](ch[i])
            return

        if k == '^':
            self.cur = (self.cur - 1) % len(self.items)
        elif k == 'v':
            self.cur = (self.cur + 1) % len(self.items)
        elif k == '#':
            if 'action' in it:
                it['action']()
            elif 'set' in it:
                self.edit = True

        if self.cur < self.top:
            self.top = self.cur
        elif self.cur >= self.top + self.visible:
            self.top = self.cur - self.visible + 1

    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            return '{:.2f}'.format(v)
        return str(v)

    def _draw(self):
        lines = []
        end = min(self.top + self.visible, len(self.items))
        for i in range(self.top, end):
            it = self.items[i]
            mark = ('*' if (self.edit and i == self.cur)
                    else '>' if i == self.cur else ' ')
            if 'get' in it:
                line = '{}{}:{}'.format(mark, it['name'], self._fmt(it['get']()))
            else:
                line = '{}{}'.format(mark, it['name'])
            lines.append(line)
        self.oled.show_lines(*lines, sizes=self.size, padding=2)
