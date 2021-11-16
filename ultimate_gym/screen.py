from yuzulib import Screen as YuzuScreen

class Screen(YuzuScreen):
    def __init__(self):
        super().__init__(self._callback, fps=2)
        self.current_frame = None
        self.current_fps = 0

    def _callback(self, frame, fps):
        #print("callback!", frame[0][0], fps)
        self.current_frame = frame
        self.current_fps = fps

    def get(self):
        return self.current_frame, self.current_fps
