import cv2
import os
from ultralytics import YOLO

# model = YOLO("yolov8n.pt")
# model.predict("example2.mp4", show=True)

def save_img_results(dir_name: str, fname: str):
    fnames = os.listdir(dir_name)

    for fname in fnames:
        img = cv2.imread(f"{dir_name}/{fname}")
        pred = model.predict(img)[0]
        plot = pred.plot()
        cv2.imwrite(f"{fname}/{fname}", plot)

if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    dir_name = "datasets/test/images"
    fname = "validation/images"
    
    save_img_results(dir_name, fname)
