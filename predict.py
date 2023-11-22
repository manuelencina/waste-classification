from models.recyclable_object_detector import RecyclableObjectDetector

def main():
    model = RecyclableObjectDetector(path="runs/detect/train/weights/best.pt")
    conf = {"source": 0, "stream":True, "device":'cpu'}
    model.execute(conf)

if __name__ == "__main__":
    main()