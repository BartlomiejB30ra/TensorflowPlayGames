import time

import numpy as np
from grabscreen import grab_screen, process_img
from get_keys import key_check
import os


class CreateTrainingData:
    PAUSE_TIMEOUT = 2

    def __init__(self, file_name):
        self.file_name = file_name
        self.training_data = self.init_training_data()
        self.pause = True
        self.last_pause_time = 0

    def init_training_data(self):
        if os.path.isfile(self.file_name):
            print('File exists, loading previous data!')
            return list(np.load(self.file_name))
        else:
            print('File does not exist, starting fresh!')
            return []

    def keys_to_output(self, keys):
        # [J,Z,L] boolean values.
        output = [0, 0, 0]

        if 'J' in keys:
            output[0] = 1
        elif 'L' in keys:
            output[2] = 1
        elif 'Z' in keys:
            output[1] = 1
        if 'P' in keys:
            self.toggle_pause()
        return output

    def start_collecting_data(self):
        while True:
            screen = grab_screen()
            # print('Frame took {} seconds'.format(time.time() - last_time))
            # last_time = time.time()
            processed_img = process_img(screen)
            keys = key_check()
            output = self.keys_to_output(keys)
            if not self.pause:
                self.training_data.append([processed_img, output])
                if len(self.training_data) % 500 == 0:
                    print(len(self.training_data))
                    np.save(self.file_name, self.training_data)

    def toggle_pause(self):
        current_time = time.time()
        if current_time - self.last_pause_time > self.PAUSE_TIMEOUT:
            self.last_pause_time = current_time
            self.pause = not self.pause
            print("Pause {}".format(self.pause))


if __name__ == "__main__":
    ctd = CreateTrainingData('training_data.npy')
    ctd.start_collecting_data()
