import time
import random
import argparse
import os
from pathlib import Path
import cv2
import numpy as np

from mss import mss
mon = {'left': 0, 'top': 0, 'width': 200, 'height': 500}
data_path = Path(os.path.dirname(__file__)).resolve()
with mss() as sct:
    img = sct.grab(mon)
    img = np.array(img)
    print(img.shape)
    cv2.imwrite("{}/screen.png".format(str(data_path)), img)
