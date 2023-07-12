from ultralytics import YOLO
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "data.yaml")
    model = YOLO("yolov8n.pt")

    model.train(data=data_file, epochs=50, imgsz=640, batch=8, optimizer="Adam", device=0)