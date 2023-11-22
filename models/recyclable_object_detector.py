from ultralytics import YOLO
import time
import numpy as np


class RecyclableObjectDetector:
    def __init__(self, path: str):
        self.model = YOLO(path)

    def execute(self, conf: dict):
        results = self.model(**conf)
        while True:
            for result in results:
                is_cardboard = "isNotCARBOARD"
                probs = result.boxes.conf
                if probs.nelement() == 1 and probs.item() >= 0.6:
                    is_cardboard = "isCARBOARD"
                    print("Prob:", probs.item())
                    time.sleep(5)
                np.savetxt('output.txt', np.array([is_cardboard]), fmt='%s')