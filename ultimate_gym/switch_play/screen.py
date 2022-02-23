import threading
import time
import cv2
import numpy as np

class VideoCaptureScreen:
    def __init__(self, callback, fps=60, width=980, height=500, disable_warning=False) -> None:
        self.callback = callback
        self.fps = fps
        self.disable_warning = disable_warning
        self.capture = cv2.VideoCapture('/dev/video0')
        self.capture.set(3, width)
        self.capture.set(4, height)
        self.alive = True

    def run(self):
        thread = threading.Thread(target=self.start_capture)
        thread.start()

    def start_capture(self): 
        start = time.time()
        while self.alive:
            ret, frame = self.capture.read()
            #cv2.imshow('frame',frame)
            frame = np.array(frame)
            if cv2.waitKey(33) & 0xFF in (ord('q'), 27):
                break
            elapsed_time = time.time() - start
            if self.fps:
                target_elapsed_time = 1/self.fps
                if target_elapsed_time < elapsed_time:
                    if not self.disable_warning:
                        print("warning: low fps {}".format(1/elapsed_time))
                else:
                    time.sleep(target_elapsed_time-elapsed_time)
                elapsed_time = time.time() - start
            fps = 1/elapsed_time
            start = time.time()
            self.callback(frame, fps)
        self.capture.release()
        cv2.destroyAllWindows()

    def close(self):
        self.alive = False

class Screen:
    def __init__(self, fps=60, width=980, height=500, disable_warning=False) -> None:
        self.fps = fps
        self.current_frame = None
        self.current_fps = 0
        self.screen = VideoCaptureScreen(callback=self._callback, fps=fps, width=width, height=height, disable_warning=disable_warning)
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