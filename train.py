from ultralytics import YOLO
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "data.yaml")
    model = YOLO("yolov8n.pt")

    model.train(data=data_file, epochs=10, imgsz=(416,416), batch=8, optimizer="Adam")