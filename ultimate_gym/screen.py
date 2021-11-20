from libultimate import Screen as YuzuScreen
from threading import (Event)

class Screen(YuzuScreen):
    def __init__(self, fps=60):
        super().__init__(self._callback, fps=fps)
        self.current_frame = None
        self.current_fps = 0
        self.event = Event()

    def _callback(self, frame, fps):
        #print("callback!", frame[0][0], fps)
        self.current_frame = frame
        self.current_fps = fps
        self.event.set()

    def get(self):
        self.event.wait()
        self.event.clear()
        return self.current_frame, self.current_fps
