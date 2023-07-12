from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("runs/detect/train/weights/best.pt")
    model.predict(
        "videoname.mp4",
        conf=0.6,
        max_det=10,
        device=0,
        save=True,
        show=True
    )