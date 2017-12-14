import numpy as np
from grabscreen import grab_screen, process_img
import cv2
import time
from direct_keys import PressKey, ReleaseKey, Z, J, L
from alexnet import alexnet

WIDTH = 48
HEIGHT = 37
LR = 1e-3  # lerning rate
EPOCHS = 20
MODEL_NAME = 'mario-kart-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2', EPOCHS)


def straight():
    PressKey(Z)
    ReleaseKey(J)
    ReleaseKey(L)


def left():
    PressKey(Z)
    PressKey(J)
    ReleaseKey(L)


def right():
    PressKey(Z)
    PressKey(L)
    ReleaseKey(J)


model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)


if __name__ == "__main__":
    last_time = time.time()

    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    while (True):
        # 800x600 windowed mode
        screen = grab_screen()
        # print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()
        screen = process_img(screen)
        cv2.imshow('test', screen)
        moves = list(np.around(model.predict([screen.reshape(48, 37, 1)])[0]))
        # print(moves)

        if moves == [1, 0, 0]:
            left()
        elif moves == [0, 1, 0]:
            straight()
        elif moves == [0, 0, 1]:
            right()

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
