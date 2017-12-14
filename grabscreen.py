import cv2
import numpy as np

from PIL import ImageGrab


def process_img(image):
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.resize(image_np, (48, 37))
    # processed_img = cv2.equalizeHist(simplified_img)
    return processed_img


def grab_screen():
    return np.array(ImageGrab.grab(bbox=(0, 52, 480, 372)))
