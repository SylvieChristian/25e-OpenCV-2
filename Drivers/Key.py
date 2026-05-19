from datetime import timedelta

import gpiod
from gpiod.line import Bias, Direction, Edge

DEFAULT_KEY_MAP = {
    144: '^',   # Pin 7  PAC.06
    43:  'v',   # Pin 33 PH.00
    106: '#',   # Pin 31 PQ.06
    105: '*',   # Pin 29 PQ.05
}


class Keys:
    def __init__(self, key_map=None, chip='/dev/gpiochip0', debounce_ms=30):
        self.key_map = dict(key_map) if key_map else DEFAULT_KEY_MAP
        s = gpiod.LineSettings(
            direction=Direction.INPUT,
            bias=Bias.PULL_UP,
            edge_detection=Edge.FALLING,
            debounce_period=timedelta(milliseconds=debounce_ms),
        )
        self._req = gpiod.request_lines(
            chip, consumer='keys',
            config={k: s for k in self.key_map},
        )

    def get(self, timeout_ms=None):
        td = None if timeout_ms is None else timedelta(milliseconds=timeout_ms)
        if self._req.wait_edge_events(timeout=td):
            ev = self._req.read_edge_events()[0]
            return self.key_map.get(ev.line_offset)
        return None

    def close(self):
        self._req.release()
