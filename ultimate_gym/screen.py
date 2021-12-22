from yuzulib import Screen as YuzuScreen
import threading

class Screen:
    def __init__(self, fps=60, disable_warning=False) -> None:
        self.fps = fps
        self.current_frame = None
        self.current_fps = 0
        self.screen = YuzuScreen(callback=self._callback, fps=fps, disable_warning=disable_warning)
        self.event = threading.Event()

    def run(self):
        thread = threading.Thread(target=self._run)
        thread.start()

    def _run(self):
        self.screen.run()

    def _callback(self, frame, fps):
        self.current_frame = frame
        self.current_fps = fps
        self.event.set()

    def get(self):
        self.event.wait()
        self.event.clear()
        return self.current_frame, self.current_fps