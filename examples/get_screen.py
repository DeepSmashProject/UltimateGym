import time
import random
import argparse
import os
from pathlib import Path
import cv2
import numpy as np
img = cv2.imread("./screen_0_0.png")
print(img)
#img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imwrite("./screen_create.png", img)