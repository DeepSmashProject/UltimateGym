from libultimate.client import UltimateClient
import threading

class Screen:
    def __init__(self, fps=60, address="http://localhost:6000", render=False, width=256, height=256) -> None:
        self.fps = fps
        self.render = render
        self.width = width
        self.height = height
        self.current_frame = None
        self.current_fps = 0
        self.event = threading.Event()
        self.client = UltimateClient(address=address, disable_warning=True)

    def run(self):
        thread = threading.Thread(target=self._run)
        thread.start()

    def _run(self):
        self.client.run_screen(self._callback, fps=self.fps, render=self.render, width=self.width, height=self.height)

    def _callback(self, frame, fps):
        self.current_frame = frame
        self.current_fps = fps
        self.event.set()

    def get(self):
        self.event.wait()
        self.event.clear()
        return self.current_frame, self.current_fps