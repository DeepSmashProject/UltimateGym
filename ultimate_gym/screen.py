from yuzulib import Screen as YuzuScreen

class Screen(YuzuScreen):
    def __init__(self):
        super().__init__(self._callback, fps=2)
        self.frame = None
        self.fps = 0

    def _callback(self, frame, fps):
        #print("callback!", frame[0][0], fps)
        self.frame = frame
        self.fps = fps

    def get(self):
        return self.frame, self.fps
