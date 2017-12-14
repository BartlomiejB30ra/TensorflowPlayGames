import numpy as np
import os
import sys
import tensorflow as tf

import cv2
import time

from direct_keys import PressKey, Z
from grabscreen import grab_screen, process_img

START_TIMEOUT = 5


if __name__ == "__main__":
    # last_time = time.time()
    print("starting in {}s ...".format(START_TIMEOUT))
    time.sleep(START_TIMEOUT)
    while True:
        PressKey(Z)
        screen = grab_screen()
        # print('Frame took {} seconds'.format(time.time() - last_time))
        # last_time = time.time()
        processed_img = process_img(screen)
        cv2.imshow("object detection", cv2.resize(processed_img, (480, 372), interpolation=cv2.INTER_NEAREST))
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
