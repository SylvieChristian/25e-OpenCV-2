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
    # oled: SSD1306 实例  items: 菜单项列表  visible: 可见行数  size: 字号
    def __init__(self, oled, items, visible=4, size=13):
        self.oled = oled
        self.items = items
        self.visible = visible
        self.size = size
        self.cur = 0            # 当前高亮行索引
        self.top = 0            # 滚屏时可视区域第一行
        self.edit = False       # 是否处于参数编辑模式

    # 按 name 快速读取某菜单项的当前值
    def __getitem__(self, name):
        for it in self.items:
            if it['name'] == name:
                return it['get']()
        raise KeyError(name)

    # 主循环步进：等待按键 → 处理 → 刷新显示，每步最多阻塞 timeout_ms 毫秒
    def step(self, timeout_ms=100):
        k = self.oled.wait_key(timeout_ms=timeout_ms)
        if k:
            self._handle(k)
        self._draw()

    # 按键分发：编辑模式下调值/切换选项，浏览模式下移动光标或触发动作
    def _handle(self, k):
        it = self.items[self.cur]
        if self.edit:
            if k in ('#', '*'):
                self.edit = False                                 # # 或 * 退出编辑
            elif 'range' in it:
                lo, hi, step = it['range']
                v = it['get']()
                if k == '^': v = min(hi, v + step)                # 上调，不超上限
                if k == 'v': v = max(lo, v - step)                # 下调，不低下限
                it['set'](v)
            elif 'choices' in it:
                ch = it['choices']; v = it['get']()
                i = ch.index(v) if v in ch else 0
                if k == '^': i = (i + 1) % len(ch)               # 循环切换到下一选项
                if k == 'v': i = (i - 1) % len(ch)               # 循环切换到上一选项
                it['set'](ch[i])
            return

        # 浏览模式
        if k == '^':
            self.cur = (self.cur - 1) % len(self.items)
        elif k == 'v':
            self.cur = (self.cur + 1) % len(self.items)
        elif k == '#':
            if 'action' in it:
                it['action']()                                    # 触发动作项
            elif 'set' in it:
                self.edit = True                                  # 进入编辑模式

        # 滚动窗口跟随光标
        if self.cur < self.top:
            self.top = self.cur
        elif self.cur >= self.top + self.visible:
            self.top = self.cur - self.visible + 1

    # 格式化值：浮点数保留两位小数，其余直接转字符串
    @staticmethod
    def _fmt(v):
        if isinstance(v, float):
            return '{:.2f}'.format(v)
        return str(v)

    # 绘制可见窗口：逐行拼装 "标记 名称:值"，调 show_lines 渲染到屏幕
    def _draw(self):
        lines = []
        end = min(self.top + self.visible, len(self.items))
        for i in range(self.top, end):
            it = self.items[i]
            mark = ('*' if (self.edit and i == self.cur)          # * 表示正在编辑该项
                    else '>' if i == self.cur else ' ')            # > 表示当前光标
            if 'get' in it:
                line = '{}{}:{}'.format(mark, it['name'], self._fmt(it['get']()))
            else:
                line = '{}{}'.format(mark, it['name'])
            lines.append(line)
        self.oled.show_lines(*lines, sizes=self.size, padding=2)
